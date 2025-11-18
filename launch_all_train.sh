#!/bin/bash
accelerate launch --num_processes=1 sdxl_train.py --dataset_config="dataset1.toml" --config_file="config1.toml" --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 2.5 --vae_custom_scale 0.1280 --vae_custom_shift 0.1726 --use_zero_cond=True  --vae_batch_size 6 --vae_reflection_padding
echo "All training jobs finished. Press any key to close..."
read -n 1 -s -r -p ""
