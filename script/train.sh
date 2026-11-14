#!/bin/bash

# ckpt_path
ckpt_path=out/ft_csl_full/checkpont_19.pth
# output
out_path=out/ft_csl_full/checkpont_19

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --include localhost:0,1,2,3 --master_port 28411 fine_tuning_lora.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --max_length 256 \
  --lr 3e-4 \
  --finetune $ckpt_path \
  --output_dir $output_dir \
  --dataset CSL_Daily \
  --lambda_lex 0.4