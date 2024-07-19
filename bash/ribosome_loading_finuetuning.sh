# -*- coding: utf-8 -*-
seed=2024
mkdir -p '/home/yunlongli/code/UTR-LM/output/mrl/logs'
CUDA_VISIBLE_DEVICES="0,1" python3 \
    train_ribosome_loading.py \
    --pretrained_rinalmo_weights ./output/pretrain_v1/utr_lm/pretrain-v1/checkpoints/epoch46-step11750-loss=1.199.ckpt \
    --output_dir ./output/mrl/ \
    --accelerator gpu \
    --max_epochs 30 \
    --batch_size 1024 \
    --num_workers 8 \
    --wandb \
    --wandb_version v1 \
    --checkpoint_every_epoch \
    --wandb_experiment_name mrl \
    --wandb_project mrl \
    --data_dir ./data \
    >> ./output/mrl/logs/train_exp2.log 2>&1
    #--yaml_config 'ft_schedules/pretrain.yaml'\
    #--test_only \
    #--init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt \
    #--data_dir ./ss_data \
    #--prepare_data \