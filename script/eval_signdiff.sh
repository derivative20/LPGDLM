#!/bin/bash

# ckpt_path
ckpt_path=out/ft_csl_full/checkpont_19.pth
# output
out_path=out/ft_csl_full/checkpont_19

deepspeed --include localhost:0 --master_port 28411 fine_tuning_lora.py \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --epochs 20 \
    --opt AdamW \
    --lr 3e-4 \
    --output_dir $out_path \
    --finetune $ckpt_path \
    --dataset CSL_Daily \
    --eval \
