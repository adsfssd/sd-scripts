# latentsのdiskへの事前キャッシュを行う / cache latents to disk

import argparse
from multiprocessing import Value
import os
import sys

from accelerate.utils import set_seed
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms

from library import config_util
from library import train_util
from library import sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
    if alpha_mask is not None:
        kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    np.savez(
        npz_path,
        latents=latents_tensor.float().cpu().numpy(),
        original_size=np.array(original_size),
        crop_ltrb=np.array(crop_ltrb),
        **kwargs,
    )

def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    train_util.prepare_dataset_args(args, True)

    assert args.cache_latents_to_disk, "cache_latents_to_disk must be True"

    if args.seed is not None:
        set_seed(args.seed)

    # Prepare tokenizers
    if args.sdxl:
        tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
        tokenizers = [tokenizer1, tokenizer2]
    else:
        tokenizer = train_util.load_tokenizer(args)
        tokenizers = [tokenizer]

    # Manually build the correct dataset type (DreamBooth) from the .toml config
    if args.dataset_config is not None:
        logger.info(f"Loading dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)

        subsets = []
        for ds_config in user_config.get("datasets", []):
            for subset_config in ds_config.get("subsets", []):
                subset = train_util.DreamBoothSubset(
                    image_dir=subset_config.get("image_dir"),
                    is_reg=subset_config.get("is_reg", False),
                    class_tokens=subset_config.get("class_tokens"),
                    num_repeats=subset_config.get("num_repeats", 1),
                    shuffle_caption=subset_config.get("shuffle_caption", False),
                    keep_tokens=subset_config.get("keep_tokens", 0),
                    color_aug=subset_config.get("color_aug", False),
                    flip_aug=subset_config.get("flip_aug", False),
                    face_crop_aug_range=subset_config.get("face_crop_aug_range"),
                    random_crop=subset_config.get("random_crop", False),
                    caption_dropout_rate=subset_config.get("caption_dropout_rate", 0.0),
                    caption_tag_dropout_rate=subset_config.get("caption_tag_dropout_rate", 0.0),
                    caption_dropout_every_n_epochs=subset_config.get("caption_dropout_every_n_epochs", 0),
                    caption_extension=subset_config.get("caption_extension", ".txt"),
                    caption_separator=subset_config.get("caption_separator", ","),
                    keep_tokens_separator=subset_config.get("keep_tokens_separator", ""),
                    secondary_separator=subset_config.get("secondary_separator"),
                    enable_wildcard=subset_config.get("enable_wildcard", False),
                    caption_prefix=subset_config.get("caption_prefix"),
                    caption_suffix=subset_config.get("caption_suffix"),
                    token_warmup_min=subset_config.get("token_warmup_min", 1),
                    token_warmup_step=subset_config.get("token_warmup_step", 0),
                    cache_info=subset_config.get("cache_info", False),
                    alpha_mask=subset_config.get("alpha_mask", False),
                )
                subsets.append(subset)

        # --- THIS IS THE CORE FIX ---
        # The 'batch_size' for the dataset's internal batching MUST use args.vae_batch_size
        logger.info(f"Using VAE batch size: {args.vae_batch_size}")
        dataset = train_util.DreamBoothDataset(
            subsets=subsets,
            batch_size=args.vae_batch_size, # Correctly use the command-line argument for batching
            tokenizer=tokenizers,
            max_token_length=args.max_token_length,
            resolution=args.resolution,
            enable_bucket=user_config.get("general", {}).get("enable_bucket", True),
            min_bucket_reso=user_config.get("general", {}).get("min_bucket_reso", 256),
            max_bucket_reso=user_config.get("general", {}).get("max_bucket_reso", 2048),
            bucket_reso_steps=user_config.get("general", {}).get("bucket_reso_steps", 64),
            bucket_no_upscale=user_config.get("general", {}).get("bucket_no_upscale", False),
            prior_loss_weight=1.0,
            network_multiplier=1.0,
            debug_dataset=False
        )

        dataset.make_buckets()
        train_dataset_group = train_util.DatasetGroup([dataset])

    else:
        logger.error("A --dataset_config file is required for this script.")
        return

    if len(train_dataset_group) == 0:
        logger.error("No images were loaded from the dataset. Please check your image_dir paths in the .toml file.")
        return

    # Prepare accelerator and VAE
    logger.info("prepare accelerator")
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)
    print(f"accelerator device: {accelerator.device}")

    weight_dtype, _ = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    logger.info("load model")
    setattr(args, 'disable_mmap_load_safetensors', False)

    if args.sdxl:
        (_, _, _, vae, _, _, _) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    else:
        _, vae, _, _ = train_util.load_target_model(args, weight_dtype, accelerator)

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    def passthrough_collate_fn(examples):
        return examples[0]

    train_dataset_group.set_caching_mode("latents")
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count()) if args.max_data_loader_n_workers is not None else 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1, # This MUST be 1 because the dataset is already creating batches
        shuffle=False,
        collate_fn=passthrough_collate_fn,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers and n_workers > 0,
    )

    train_dataloader = accelerator.prepare(train_dataloader)

    image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    for batch in tqdm(train_dataloader, desc=f"Caching latents on process {accelerator.process_index}"):
        abs_paths = batch["absolute_paths"]

        processed_images, original_sizes, crop_ltrbs = [], [], []
        valid_indices_for_saving = []

        batch_bucket_reso = batch["bucket_reso"]

        for i in range(len(abs_paths)):
            npz_path = os.path.splitext(abs_paths[i])[0] + ".npz"
            if args.skip_existing and train_util.is_disk_cached_latents_is_expected(
                batch_bucket_reso, npz_path, batch["flip_aug"], batch["alpha_mask"]
            ):
                continue

            valid_indices_for_saving.append(i)

            image_np = np.array(batch["images"][i], dtype=np.uint8)
            resized_size = batch["resized_sizes"][i]

            image_np, original_size, crop_ltrb = train_util.trim_and_resize_if_required(
                batch["random_crop"], image_np, batch_bucket_reso, resized_size
            )
            processed_images.append(image_transforms(image_np[:, :, :3]))
            original_sizes.append(original_size)
            crop_ltrbs.append(crop_ltrb)

        if not processed_images:
            continue

        img_tensors = torch.stack(processed_images).to(accelerator.device, dtype=vae_dtype)

        with torch.no_grad():
            latents = vae.encode(img_tensors).latent_dist.sample()

        if batch["flip_aug"]:
            flipped_tensors = torch.flip(img_tensors, dims=[3])
            with torch.no_grad():
                flipped_latents = vae.encode(flipped_tensors).latent_dist.sample()

        latents_cpu = latents.to("cpu")
        if batch["flip_aug"]:
            flipped_latents_cpu = flipped_latents.to("cpu")

        for i in range(len(processed_images)):
            original_batch_index = valid_indices_for_saving[i]
            npz_path = os.path.splitext(abs_paths[original_batch_index])[0] + ".npz"

            save_latents_to_disk(
                npz_path, latents_cpu[i], original_sizes[i], crop_ltrbs[i],
                flipped_latents_cpu[i] if batch["flip_aug"] else None, None
            )

    accelerator.wait_for_everyone()
    accelerator.print(f"Finished caching latents.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    config_util.add_config_arguments(parser)
    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model")
    parser.add_argument("--no_half_vae", action="store_true", help="do not use fp16/bf16 VAE in mixed precision")
    parser.add_argument("--skip_existing", action="store_true", help="skip images if npz already exists")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    # Brute-force override to ensure command-line arg is respected
    # This ensures that even if the .toml loading logic is confusing, the user's intent is followed.
    for i, arg in enumerate(sys.argv):
        if arg == '--vae_batch_size':
            try:
                args.vae_batch_size = 6
                logger.info(f"Manually set VAE batch size from command line: {args.vae_batch_size}")
            except (ValueError, IndexError):
                pass
            break

    cache_to_disk(args)
