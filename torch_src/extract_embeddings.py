# extract_embeddings.py
import os
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from diffusion_utils import DiffusionUtils
from config import MNIST_config as config
from diffusion_transformer import get_diffution_transformer  # or wherever DiT lives

import numpy as np
from config import MNIST_config as config  # your existing config
from diffusion_transformer import *
SEED = 42

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

@torch.no_grad()
def extract_dataset_embeddings(
    model: nn.Module,
    mlp: nn.Module | None,
    diff_utils: DiffusionUtils,
    dataloader: DataLoader,
    layer_idx: int,
    noise_timestep: int,
    device: torch.device,
    avg_embeddings: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        embeddings: (N_samples, d_model)
        labels    : (N_samples,)
    """
    model.eval()

    all_embs = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        B = images.size(0)
        timesteps = torch.full(
            (B,), fill_value=noise_timestep,
            device=device, dtype=torch.long
        )
        
        noisy_image_timestep, _ = diff_utils.noisy_it(images, timesteps)
        x_t = noisy_image_timestep["noisy_images"]   # (B, C, H, W)
        t = noisy_image_timestep["timesteps"]        # (B,)

        # token features after layer_idx blocks: (B, N, d_model)
        tokens = model.forward_until_layer(x_t, t, labels, layer_idx)
        
        if mlp:
            tokens = mlp(tokens)
            
        # average over tokens N -> (B, d_model)
        if avg_embeddings:
            emb = tokens.mean(dim=1)
        else:
            emb = tokens

        all_embs.append(emb.cpu())
        all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels

def get_fixed_mnist_loaders(
    subset_dir: str = "./fixed_mnist_subset",
    batch_size: int = None,
):
    if batch_size is None:
        batch_size = config.batch_size

    # load npy files
    train_imgs = np.load(f"{subset_dir}/train_images_1k.npy")   # (1000, 1, 28, 28)
    train_labels = np.load(f"{subset_dir}/train_labels_1k.npy") # (1000,)
    test_imgs = np.load(f"{subset_dir}/test_images_1k.npy")     # (1000, 1, 28, 28)
    test_labels = np.load(f"{subset_dir}/test_labels_1k.npy")   # (1000,)

    # to tensors
    train_imgs_t = torch.from_numpy(train_imgs).float()
    train_labels_t = torch.from_numpy(train_labels).long()
    test_imgs_t = torch.from_numpy(test_imgs).float()
    test_labels_t = torch.from_numpy(test_labels).long()

    train_ds = TensorDataset(train_imgs_t, train_labels_t)
    test_ds  = TensorDataset(test_imgs_t,  test_labels_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,   # fixed order, always the same
        num_workers=0,   # everything already in memory
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def main():
    set_seed(SEED)
    device = config.device

    # ---- config for this run ----
    out_dir = "./embeddings_baseline_fullembed_50_000"
    os.makedirs(out_dir, exist_ok=True)

    # ---- model ----
    model = DiT(config).to(device)
    mlp = RepProjectionHead(config.d_model).to(device)

    # load trained diffusion model checkpoint

    ckpt_path = "./checkpoints/mnist_baseline_50_000.pt"  # adjust to your path
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])  # or state if you saved directly
    mlp.load_state_dict(state['rep_head'])
    
    mlp = None

    diff_utils = DiffusionUtils(config)

    train_loader, test_loader = get_fixed_mnist_loaders(
        subset_dir="./fixed_mnist_subset",
        batch_size=config.batch_size,
    )
    
    layer_idxs = [1,2,3,4,5,6]
    noise_timesteps = list(range(1, 400, 5))
    
    for layer_idx in layer_idxs:
        for noise_timestep in noise_timesteps:
            # ---- extract embeddings ----
            print("Extracting train embeddings...")
            train_emb, train_labels = extract_dataset_embeddings(
                model=model,
                mlp=mlp,
                diff_utils=diff_utils,
                dataloader=train_loader,
                layer_idx=layer_idx,
                noise_timestep=noise_timestep,
                device=device,
                avg_embeddings=False,
            )

            print("Extracting test embeddings...")
            test_emb, test_labels = extract_dataset_embeddings(
                model=model,
                mlp=mlp,
                diff_utils=diff_utils,
                dataloader=test_loader,
                layer_idx=layer_idx,
                noise_timestep=noise_timestep,
                device=device,
                avg_embeddings=False,
            )

            d_model = train_emb.shape[1]

            # ---- save ----
            train_path = os.path.join(out_dir, f"mnist_train_layer{layer_idx}_t{noise_timestep}.pt")
            test_path  = os.path.join(out_dir, f"mnist_test_layer{layer_idx}_t{noise_timestep}.pt")

            torch.save(
                {
                    "embeddings": train_emb,    # (N_train, d_model)
                    "labels": train_labels,     # (N_train,)
                    "layer_idx": layer_idx,
                    "noise_timestep": noise_timestep,
                    "d_model": d_model,
                },
                train_path,
            )

            torch.save(
                {
                    "embeddings": test_emb,
                    "labels": test_labels,
                    "layer_idx": layer_idx,
                    "noise_timestep": noise_timestep,
                    "d_model": d_model,
                },
                test_path,
            )

            print(f"Saved train embeddings to {train_path}")
            print(f"Saved test embeddings to {test_path}")


if __name__ == "__main__":
    main()
