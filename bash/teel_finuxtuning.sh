# -*- coding: utf-8 -*-
mkdir -p '/home/yunlongli/code/utr/output/teel/logs'
CUDA_VISIBLE_DEVICES="2,3" python3 \
    train_te_el.py \
    --task_type TE \
    --pretrained_rinalmo_weights ./output/pretrain_v1/utr_lm/pretrain-v1/checkpoints/epoch04-step40000-loss=1.176.ckpt \
    --output_dir ./output/teel/ \
    --accelerator gpu \
    --max_epochs 10 \
    --batch_size 64 \
    --num_workers 8 \
    --seed 2024 \
    --wandb \
    --wandb_version v2 \
    --checkpoint_every_epoch \
    --wandb_experiment_name teel \
    --wandb_project teel \
    --data_dir ./utr/data/downstream/teel/Muscle_sequence.csv \
    >> ./output/teel/logs/train_exp2.log 2>&1
    #--yaml_config 'ft_schedules/pretrain.yaml'\
    #--test_only \
    #--init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt \
