# -*- coding: utf-8 -*-
#export PYTHONPATH=/home/yunlongli/code/RiNALMo:$PYTHONPATH

seed=2024
mkdir -p '/home/yunlongli/code/RiNALMo/output/sec_struct_prediction/logs'
CUDA_VISIBLE_DEVICES="2,3" python3 \
    train_sec_struct_prediction.py \
    --pretrained_rinalmo_weights ./output/pretrain_v1/utr_lm/pretrain-v1/checkpoints/epoch04-step40000-loss=1.176.ckpt \
    --dataset archiveII_5s \
    --output_dir ./output/sec_struct_prediction \
    --accelerator gpu \
    --data_dir ./rinalmo/data/downstream/secondary_structure/ss_data \
    --max_epochs 5 \
    --wandb \
    --wandb_version v2 \
    --wandb_experiment_name ss \
    --wandb_project ss \
    >> ./output/sec_struct_prediction/logs/train_exp2.log 2>&1
    #--yaml_config 'ft_schedules/pretrain.yaml'\
    #--test_only \
    #--init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt \
    #--data_dir ./ss_data \
    #--prepare_data \
