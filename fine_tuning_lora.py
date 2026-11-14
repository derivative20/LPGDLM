import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models_lexical import Uni_Sign
import utils as utils
from datasets_lexical import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
from peft import LoraConfig, get_peft_model
from SLT_metrics import translation_performance
from transformers import get_scheduler
from config import *

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating model:")
    model = Uni_Sign(args=args)
    model.train()

    print(f"Creating dataset:")
    
    if "CSL" in args.dataset:
        lang = 'Chinese'
    else:
        lang = 'English'

    if args.dataset == "CSL_Daily":   
        train_data = S2T_Dataset(path=train_ds_path[args.dataset], args=args, phase='train', tokenizer=model.llada_tokenizer, language=lang)   
    elif args.dataset == "How2Sign":
        train_data = S2T_Dataset(path=train_ds_path[args.dataset],
                        args=args, phase='train', tokenizer=model.llada_tokenizer, language=lang)  
    else:
        raise NotImplementedError

    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn_train,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)
        
    if args.dataset == "CSL_Daily":   
        test_data = S2T_Dataset(path=test_ds_path[args.dataset], 
                        args=args, phase='test', tokenizer=model.llada_tokenizer, language=lang)   
    elif args.dataset == "How2Sign":
        test_data = S2T_Dataset(path=test_ds_path[args.dataset], 
                        args=args, phase='test', tokenizer=model.llada_tokenizer, language=lang)  
    else:
        raise NotImplementedError

    print(test_data)

    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    if "How2Sign" not in args.dataset:
        dev_data = S2T_Dataset(path=dev_ds_path[args.dataset], 
                    args=args, phase='test', tokenizer=model.llada_tokenizer, language=lang)
        print(dev_data)

        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=dev_sampler,
                                    pin_memory=args.pin_mem)
    else:
        dev_dataloader = test_dataloader

    print('***********************************')
    print('Applying LoRA to llada_model...')
    print('***********************************')
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model.llada_model = get_peft_model(model.llada_model, peft_config=lora_config)
    
    # Load checkpoint after LoRA injection
    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    
    model_without_ddp = model

    if args.eval:
        # model to bf16
        model = model.to(torch.bfloat16).to("cuda")

        if utils.is_main_process():
            if "How2Sign" not in args.dataset:
                print("📄 dev result")
                evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        return 

    # LLaDA: only LoRA params trainable; everything else: full fine-tuning
    for name, param in model.named_parameters():
        param.requires_grad = True
    for name, param in model.llada_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    lora_param_count = sum(p.numel() for n, p in model.llada_model.named_parameters() if p.requires_grad) / 1e6
    print(f'number of params: {n_parameters}M, trainable: {trainable_params}M (LoRA in LLaDA: {lora_param_count}M)')

    # Separate param groups: LoRA params use seperate learning rate
    llada_params = [p for n, p in model.llada_model.named_parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith('llada_model.')]

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': llada_params, 'lr': args.llada_lr},
    ], weight_decay=args.weight_decay)
    
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        # eval
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        if max_accuracy < test_stats["bleu4"]:
            max_accuracy = test_stats["bleu4"]
        
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} test videos: {test_stats['bleu4']:.2f}")
        print(f'Max BLEU-4: {max_accuracy:.2f}')

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_path)

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
                        
            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# def get_trainable_state_dict(model):
#     """Save only trainable params (LoRA adapters + non-llada full weights)."""
#     trainable_names = set()
#     for name, param in model.named_parameters():
#         if param.requires_grad or "llada" not in name:
#             trainable_names.add(name)
#             # Mirror left <-> right for weight sharing
#             if 'left' in name:
#                 trainable_names.add(name.replace('left', 'right'))
#             # if 'right' in name:
#             #     trainable_names.add(name.replace('right', 'left'))
#     return {k: v for k, v in model.state_dict().items() if k in trainable_names}

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        stack_out = model(src_input, tgt_input, "train")
        
        total_loss = stack_out['total_loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=stack_out['loss_main'].item())
        metric_logger.update(loss_lexical=stack_out['loss_lexical'].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()
    model_without_ddp.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        tgt_name = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            stack_out = model_without_ddp(src_input, tgt_input, "test")

            if "CSL" in args.dataset:
                output = model_without_ddp.generate(stack_out, max_new_tokens=32)
            else:
                output = model_without_ddp.generate(stack_out, max_new_tokens=64)

            # Decode generated token IDs
            decoded = model_without_ddp.llada_tokenizer.batch_decode(output, skip_special_tokens=True)
            for i in range(len(decoded)):
                tgt_pres.append(decoded[i])
                tgt_refs.append(tgt_input['gt_text'][i])
                tgt_name.append(src_input['name_batch'][i])

    # Character-level spacing for Chinese evaluation
    if args.dataset == 'CSL_Daily':
        tgt_pres = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    try:
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
    except ValueError as e:
        bleu_dict = {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        rouge_score = 0.0
    for k,v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        result = []
        for i in range(len(tgt_name)):
            result.append({
                "sample": tgt_name[i],
                "ref": tgt_refs[i],
                "pred": tgt_pres[i],
            })
        
        res = {"bleu1": bleu_dict['bleu1'], "bleu2": bleu_dict['bleu2'], "bleu3": bleu_dict['bleu3'], "bleu4": bleu_dict['bleu4'], "RougeL": rouge_score, "detail": result}
        res_filename = args.output_dir+f'/{phase}_lora_result.json'
        with open(res_filename,'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj", "ff_out", "ff_proj", "up_proj"], help='LoRA target modules')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=128, help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lambda_lex', type=float, default=0.1, help='rate of the lexical loss')
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)