# r2bc/train_joint_bc_lowdim.py

import os
import argparse
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from r2bc.datasets.joint_bc_lowdim_dataset import JointBCLowDimDataset


class JointBCLowDimMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 18, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--task", type=str, default="bimanual_lift_long_block")
    parser.add_argument("--save-dir", type=str, default="checkpoints/joint_bc_lowdim")
    parser.add_argument("--max-episodes", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = JointBCLowDimDataset(
        data_root=args.data_root,
        task_name=args.task,
        max_episodes=args.max_episodes,
        num_keyframes=2,
    )

    print("[train joint bc] dataset summary:", dataset.summary())

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = JointBCLowDimMLP(
        input_dim=dataset.x_dim,
        output_dim=dataset.y_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                pred = model(x)
                loss = loss_fn(pred, y)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"[epoch {epoch:04d}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val:
            best_val = val_loss

            ckpt = {
                "model_state": model.state_dict(),
                "input_dim": dataset.x_dim,
                "output_dim": dataset.y_dim,
                "hidden_dim": args.hidden_dim,
                "x_mean": dataset.x_mean,
                "x_std": dataset.x_std,
                "y_mean": dataset.y_mean,
                "y_std": dataset.y_std,
                "task": args.task,
                "num_keyframes": dataset.num_keyframes,
                "dataset_summary": dataset.summary(),
                "best_val_loss": best_val,
            }

            save_path = join(args.save_dir, "best.pt")
            torch.save(ckpt, save_path)

    print("[train joint bc] done")
    print("[train joint bc] best_val_loss:", best_val)
    print("[train joint bc] saved:", join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()