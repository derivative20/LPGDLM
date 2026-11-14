output_dir=out_pretrain/pretrain_csl_full

deepspeed --include localhost:0,1,2,3 --master_port 28411 pre_training.py \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --dataset CSL_News \
  --lambda_anchor 0.1