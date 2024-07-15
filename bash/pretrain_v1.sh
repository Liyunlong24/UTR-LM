# -*- coding: utf-8 -*-
#DO NOTE EXECUTE THIS BASH SCRIPT DIRECTLY!!!

# on 32019 server
#source activate abdev
export PYTHONPATH=/home/yunlongli/code/utr:$PYTHONPATH

work_dir="/home/yunlongli/code/utr/output"

config_name="pretrain_v1"
seed=2024
output_dir=$work_dir/${config_name}
mkdir -p $output_dir/logs
ulimit -n 4096 && \
CUDA_VISIBLE_DEVICES="2,3" python3 \
    rinalmo/train.py \
    --yaml_config './ft_schedules/pretrain.yaml'\
    --output_dir $output_dir \
    --log_lr \
    --max_epochs 5 \
    --wandb \
    --wandb_version v1 \
    --wandb_project utr_lm \
    --experiment_name utr_lm \
    --seed $seed \
    --checkpoint_every_epoch \
    --save_top_k 10 \
    --use_ema true \
    >> $output_dir/logs/train_exp2.log 2>&1
    #--replace_sampler_ddp=True \
    #--wandb_entity lyl \
  #--gpus 2 \
    #--precision 32 \
   # --log_every_n_steps 8 \
   # >> $output_dir/logs/train_exp2.log 2>&1
