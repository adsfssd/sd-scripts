#!/bin/bash
accelerate launch sdxl_train.py --dataset_config="/home/bluvoll/caching.toml" --config_file="config1.toml" --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 2.5 --vae_custom_scale 0.1280 --vae_custom_shift 0.1726 --use_zero_cond=True  --vae_batch_size 6 --vae_reflection_padding --freeze_unet_blocks 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --deepspeed 
echo "All training jobs finished. Press any key to close..."
read -n 1 -s -r -p ""
