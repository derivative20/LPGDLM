import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models_pretrain import Uni_Sign
import utils as utils
from datasets_pretrain import S2T_Dataset
from datasets_pretrain import S2T_Dataset_news
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
from timm.optim import create_optimizer  # noqa: F401 (kept for compatibility)
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

    if args.dataset == "CSL_News":    
        train_data = S2T_Dataset_news(path=train_ds_path[args.dataset], 
                    args=args, phase='train', tokenizer=model.llada_tokenizer, language=lang)   
    elif args.dataset == "OpenASL":
        train_data = S2T_Dataset(path=train_ds_path[args.dataset], 
                    args=args, phase='train', tokenizer=model.llada_tokenizer, language=lang)
    elif args.dataset == "YoutubeASL":
        train_data = S2T_Dataset_news(path=train_ds_path[args.dataset], 
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

    if args.dataset == "CSL_News":
        dev_data = S2T_Dataset_news(path=dev_ds_path[args.dataset], 
                    args=args, phase='test', tokenizer=model.llada_tokenizer, language=lang)
    elif args.dataset == "OpenASL":
        dev_data = S2T_Dataset(path=dev_ds_path[args.dataset], 
                    args=args, phase='test', tokenizer=model.llada_tokenizer, language=lang)
    elif args.dataset == "YoutubeASL":
        dev_data = S2T_Dataset_news(path=dev_ds_path[args.dataset]", 
                    args=args, phase='test', tokenizer=model.llada_tokenizer, language=lang)
    else:
        raise NotImplementedError
    print(dev_data)

    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler,
                                pin_memory=args.pin_mem)
    
    # Load checkpoint
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
        # model转bf16
        model = model.to(torch.bfloat16).to("cuda")

        if utils.is_main_process():
            if "How2Sign" not in args.dataset:
                print("📄 dev result")
                evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        return 

    # freeze all llada_model part
    for name, param in model.named_parameters():
        param.requires_grad = True
    for name, param in model.llada_model.named_parameters():
        param.requires_grad = False

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'number of params: {n_parameters}M, trainable: {trainable_params}M')

    optimizer = create_optimizer(args, model_without_ddp)
    
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
    
    # if args.eval:
    #     if utils.is_main_process():
    #         if args.task != "ISLR" and "How2Sign" not in args.dataset:
    #             print("📄 dev result")
    #             evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
    #         print("📄 test result")
    #         evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

    #     return 
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_path)

        # eval 代码
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        if args.task == "SLT":
            if max_accuracy < test_stats["bleu4"]:
                max_accuracy = test_stats["bleu4"]
            
            print(f"BLEU-4 of the network on the {len(dev_dataloader)} test videos: {test_stats['bleu4']:.2f}")
            print(f'Max BLEU-4: {max_accuracy:.2f}')

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


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 20
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
        metric_logger.update(loss_anchor=stack_out['loss_anchor'].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

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
        
            output = model_without_ddp.generate(stack_out, max_new_tokens=64)
            
            # print(output)

            # Decode generated token IDs
            decoded = model_without_ddp.llada_tokenizer.batch_decode(output, skip_special_tokens=True)
            for i in range(len(decoded)):
                tgt_pres.append(decoded[i])
                tgt_refs.append(tgt_input['gt_text'][i])
                tgt_name.append(src_input['name_batch'][i])

    # Character-level spacing for Chinese evaluation
    if args.dataset == 'CSL_News' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
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
        res_filename = args.output_dir+f'/{phase}_result.json'
        with open(res_filename,'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    parser.add_argument('--lambda_anchor', type=float, default=0.2, help='rate of the lexical loss')
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)