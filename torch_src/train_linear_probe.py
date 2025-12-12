# train_linear_probe.py
import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from config import MNIST_config as config

import matplotlib.pyplot as plt  # NEW

SEED = 42

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def main():
    set_seed(SEED)
    device = config.device

    emb_dir = "./embeddings_baseline_50_000"
    out_dir = "./linear_probe_baseline_50_000"  # MOVED/NEW: define once
    os.makedirs(out_dir, exist_ok=True)

    layer_idxs = [1, 2, 3, 4, 5, 6]
    noise_timesteps = list(range(1, 400, 5))

    # NEW: to store final test accuracies per (layer, noise_level)
    final_test_acc = {layer_idx: [] for layer_idx in layer_idxs}

    for layer_idx in layer_idxs:
        for noise_timestep in noise_timesteps:

            train_path = os.path.join(emb_dir, f"mnist_train_layer{layer_idx}_t{noise_timestep}.pt")
            test_path  = os.path.join(emb_dir, f"mnist_test_layer{layer_idx}_t{noise_timestep}.pt")

            train_data = torch.load(train_path, map_location="cpu")
            test_data  = torch.load(test_path,  map_location="cpu")

            train_emb = train_data["embeddings"]  # (N_train, d_model)
            train_labels = train_data["labels"].long()
            test_emb = test_data["embeddings"]    # (N_test, d_model)
            test_labels = test_data["labels"].long()

            d_model = train_emb.shape[1]
            num_classes = config.num_classes

            train_ds = TensorDataset(train_emb, train_labels)
            test_ds  = TensorDataset(test_emb, test_labels)

            train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False)

            model = LinearProbe(in_dim=d_model, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
            criterion = nn.CrossEntropyLoss()

            num_epochs = 20

            history = {"train_loss": [], "train_acc": [], "test_acc": []}

            for epoch in range(1, num_epochs + 1):
                # ---- train ----
                model.train()
                total_loss = 0.0
                correct = 0
                total = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * xb.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += xb.size(0)

                train_loss = total_loss / total
                train_acc = correct / total

                # ---- eval on test ----
                model.eval()
                correct_test = 0
                total_test = 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        preds = logits.argmax(dim=1)
                        correct_test += (preds == yb).sum().item()
                        total_test += xb.size(0)

                test_acc = correct_test / total_test

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)

                print(
                    f"Epoch {epoch:03d} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Train acc: {train_acc*100:.2f}% | "
                    f"Test acc: {test_acc*100:.2f}%"
                )

            # NEW: store final test acc for this (layer, noise_timestep)
            final_test_acc[layer_idx].append((noise_timestep, test_acc))

            # ---- save model + results ----
            probe_ckpt = os.path.join(out_dir, f"linear_probe_layer{layer_idx}_t{noise_timestep}.pth")
            torch.save(model.state_dict(), probe_ckpt)

            results_path = os.path.join(out_dir, f"linear_probe_results_layer{layer_idx}_t{noise_timestep}.json")
            results = {
                "layer_idx": layer_idx,
                "noise_timestep": noise_timestep,
                "num_epochs": num_epochs,
                "train_loss": history["train_loss"],
                "train_acc": history["train_acc"],
                "test_acc": history["test_acc"],
                "train_embeddings_path": train_path,
                "test_embeddings_path": test_path,
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Saved linear probe to {probe_ckpt}")
            print(f"Saved results to {results_path}")

    # NEW: after all runs, plot "Final Test Accuracy vs Noise Level per Layer"
    plt.figure()
    for layer_idx in layer_idxs:
        # sort by noise level to get nice lines
        noise_acc_pairs = sorted(final_test_acc[layer_idx], key=lambda x: x[0])
        xs = [p[0] for p in noise_acc_pairs]
        ys = [p[1] for p in noise_acc_pairs]
        plt.plot(xs, ys, marker="o", label=f"Layer {layer_idx}")

    plt.xlabel("Noise level (t)")
    plt.ylabel("Final Test Accuracy")
    plt.title("Linear Probe Final Test Accuracy vs Noise Level per Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    fig_path = os.path.join(out_dir, "linear_probe_final_test_acc_vs_noise.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved summary plot to {fig_path}")


if __name__ == "__main__":
    main()
