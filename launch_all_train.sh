#!/bin/bash
accelerate launch sdxl_train_network.py --dataset_config="dataset1.toml" --config_file="config-example-LoCON.toml" --flow_model  --flow_use_ot  --flow_timestep_distribution uniform --flow_uniform_static_ratio 2.5 --vae_custom_scale 0.1280 --vae_custom_shift 0.1726 --use_zero_cond=True  --vae_batch_size 6 --vae_reflection_padding --use_zero_cond_dropout=True --use_sga
echo "All training jobs finished. Press any key to close..."
read -n 1 -s -r -p ""
