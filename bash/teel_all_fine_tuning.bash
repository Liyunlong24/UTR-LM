#!/bin/bash
# -*- coding: utf-8 -*-

# Create the output directory and run the TE and EL tasks for the HEK、Muscle、Pc3dataset
tasks=("TE" "EL")
datasets=("HEK" "Muscle" "Pc3")

for task in "${tasks[@]}"; do
  for dataset in "${datasets[@]}"; do

    #task_initiallr_epoch_scheduler_factor_patience_minlr
    output_file="${task}_${dataset}_2e_4_200_ReduceLROnPlateau_0.9_5_5e_6"

    data_file="./data/${dataset}_sequence.csv"

    mkdir -p "/home/yunlongli/code/UTR-LM/output/$output_file/logs"
    CUDA_VISIBLE_DEVICES="2,3" python3 train_te_el.py \
      --task_type "${task}" \
      --pretrained_rinalmo_weights ./output/pretrain_v4/checkpoints/epoch194-step3120000-loss=1.171.ckpt \
      --output_dir ./output/$output_file/ \
      --accelerator gpu \
      --max_epochs 5 \
      --batch_size 512 \
      --num_workers 8 \
      --seed 2024 \
      --lr 2e-4 \
      --wandb \
      --wandb_version $output_file \
      --wandb_project "${task}_${dataset}" \
      --checkpoint_every_epoch \
      --data_dir $dataset \
      >> ./output/$output_file/logs/train_exp2.log 2>&1
  done
done
