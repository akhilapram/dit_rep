# make_mnist_subset.py
import os
import numpy as np
import torch
from torchvision import datasets, transforms

N_SAMPLES = 1000
SEED = 123  # fix seed so indices are always the same

out_dir = "./fixed_mnist_subset"
os.makedirs(out_dir, exist_ok=True)

def main():
    # same transform as your training code
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W) -> (1, H, W), [0,255] -> [0,1]
        transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=True)  # -> [-1,1]
    ])

    trainset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    rng = np.random.default_rng(SEED)

    # choose fixed random indices (or just np.arange(N_SAMPLES) if you want the first 1k)
    train_indices = rng.choice(len(trainset), size=N_SAMPLES, replace=False)
    test_indices  = rng.choice(len(testset),  size=N_SAMPLES, replace=False)

    def get_subset(dataset, indices):
        imgs = []
        labels = []
        for idx in indices:
            x, y = dataset[idx]  # x is already transformed tensor
            imgs.append(x)       # (1, 28, 28)
            labels.append(y)
        imgs = torch.stack(imgs, dim=0)          # (N, 1, 28, 28)
        labels = torch.tensor(labels, dtype=torch.int64)  # (N,)
        return imgs, labels

    train_imgs, train_labels = get_subset(trainset, train_indices)
    test_imgs,  test_labels  = get_subset(testset,  test_indices)

    # save as .npy (on CPU, float32 for images, int64 for labels)
    np.save(os.path.join(out_dir, "train_images_1k.npy"),  train_imgs.numpy())
    np.save(os.path.join(out_dir, "train_labels_1k.npy"),  train_labels.numpy())
    np.save(os.path.join(out_dir, "test_images_1k.npy"),   test_imgs.numpy())
    np.save(os.path.join(out_dir, "test_labels_1k.npy"),   test_labels.numpy())

    # also save the indices (nice for debugging/repro)
    np.save(os.path.join(out_dir, "train_indices_1k.npy"), train_indices)
    np.save(os.path.join(out_dir, "test_indices_1k.npy"),  test_indices)

    print("Saved fixed 1k/1k MNIST subset in", out_dir)


if __name__ == "__main__":
    main()
