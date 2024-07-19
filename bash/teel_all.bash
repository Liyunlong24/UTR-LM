#!/bin/bash
# -*- coding: utf-8 -*-

# Create the output directory and run the TE and EL tasks for the HEK、Muscle、Pc3dataset
datasets=("HEK" "Muscle" "Pc3")
tasks=("te" "el")
log_dir_base='/home/yunlongli/code/UTR-LM/output'

# Common parameters
pretrained_weights="./output/pretrain_v1/utr_lm/pretrain-v1/checkpoints/epoch46-step11750-loss=1.199.ckpt"
wandb_version="v1"
for task in "${tasks[@]}"; do
  for dataset in "${datasets[@]}"; do

    output_dir="$log_dir_base/$task/$dataset"
    log_file="$output_dir/logs/train_exp2.log"
    data_file="./data/${dataset}_sequence.csv"

    mkdir -p "$output_dir/logs"

    CUDA_VISIBLE_DEVICES="0,1" python3 train_te_el.py \
      --task_type "${task^^}" \
      --pretrained_rinalmo_weights $pretrained_weights \
      --output_dir $output_dir \
      --accelerator gpu \
      --max_epochs 30 \
      --batch_size 512 \
      --num_workers 8 \
      --seed 2024 \
      --lr 1e-4 \
      --wandb \
      --wandb_version $wandb_version \
      --checkpoint_every_epoch \
      --wandb_experiment_name "${task^^}_${dataset}_v1" \
      --wandb_project "${task^^}_${dataset}_v1" \
      --data_dir $data_file \
      >> $log_file 2>&1
  done
done
