import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from typing import Union


def load_loss_log(log_path: Union[str, Path], label: str) -> pd.DataFrame:
    """
    jsonl log から update と loss を読み込む。

    Args:
        log_path: jsonl log file path
        label: "freeze" or "unfreeze" など。列名に使う。

    Returns:
        DataFrame with columns: ["update", f"{label}_loss"]
    """
    log_path = Path(log_path)
    rows = []

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "update" in obj and "loss" in obj:
                rows.append({
                    "update": obj["update"],
                    f"{label}_loss": obj["loss"],
                })

    if not rows:
        raise ValueError(f"No valid update/loss rows found in: {log_path}")

    return pd.DataFrame(rows).sort_values("update")


# =========================
# ここを自分のログファイルに合わせて変更
# =========================
unfreeze_log_path = Path("r2bc_left_1ep_unfreeze_update_100.jsonl")
freeze_log_path = Path("r2bc_left_1ep_freeze_update_100.jsonl")

out_path = Path("loss_curve_freeze_vs_unfreeze.png")


# =========================
# load
# =========================
unfreeze_df = load_loss_log(unfreeze_log_path, "unfreeze")
freeze_df = load_loss_log(freeze_log_path, "freeze")

merged = pd.merge(
    unfreeze_df,
    freeze_df,
    on="update",
    how="outer",
).sort_values("update")


# =========================
# plot
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    merged["update"],
    merged["unfreeze_loss"],
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="Unfreeze",
    color="red"
)

plt.plot(
    merged["update"],
    merged["freeze_loss"],
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="Freeze",
    color="blue"
)

plt.xlabel("Update", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.title("Training Loss Curve: Freeze vs Unfreeze", fontsize=20)
plt.legend(fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()


# =========================
# summary
# =========================
print(f"Saved: {out_path}")
print(f"unfreeze log: {unfreeze_log_path}")
print(f"freeze log:   {freeze_log_path}")

print(f"unfreeze points: {unfreeze_df['unfreeze_loss'].notna().sum()}")
print(f"freeze points:   {freeze_df['freeze_loss'].notna().sum()}")

print(
    f"unfreeze first: {unfreeze_df['unfreeze_loss'].iloc[0]:.4f}, "
    f"last: {unfreeze_df['unfreeze_loss'].iloc[-1]:.4f}, "
    f"min: {unfreeze_df['unfreeze_loss'].min():.4f}"
)

print(
    f"freeze first:   {freeze_df['freeze_loss'].iloc[0]:.4f}, "
    f"last: {freeze_df['freeze_loss'].iloc[-1]:.4f}, "
    f"min: {freeze_df['freeze_loss'].min():.4f}"
)