import json
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def mean_voxel_distance_from_log(log_path: Union[str, Path], label: str) -> pd.DataFrame:
    """
    jsonl log から update ごとの GT voxel distance を読み込む。

    distance = || pred_voxel - gt_voxel ||_2

    Args:
        log_path: jsonl log file path
        label: "freeze" or "unfreeze" など。列名に使う。

    Returns:
        DataFrame with columns: ["update", f"{label}_gt_voxel_distance"]
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

            distances = []

            # module_stats_last 内の全 sample を使う
            for s in obj.get("module_stats_last", []) or []:
                gt = s.get("gt_voxel")
                pred = s.get("pred_voxel")

                if gt is not None and pred is not None:
                    gt = np.asarray(gt, dtype=float)
                    pred = np.asarray(pred, dtype=float)
                    distances.append(float(np.linalg.norm(pred - gt)))

            # fallback: last_gt_voxel / last_pred_voxel しかない場合
            if not distances and "last_gt_voxel" in obj and "last_pred_voxel" in obj:
                gt = np.asarray(obj["last_gt_voxel"], dtype=float)
                pred = np.asarray(obj["last_pred_voxel"], dtype=float)
                distances.append(float(np.linalg.norm(pred - gt)))

            if "update" in obj and distances:
                rows.append({
                    "update": obj["update"],
                    f"{label}_gt_voxel_distance": float(np.mean(distances)),
                })

    if not rows:
        raise ValueError(f"No valid voxel distance rows found in: {log_path}")

    return pd.DataFrame(rows).sort_values("update")


# =========================
# ここを自分のログファイルに合わせて変更
# logs ディレクトリ内で実行するなら "logs/" は不要
# =========================
freeze_log_path = Path("r2bc_left_1ep_freeze_update_100.jsonl")
unfreeze_log_path = Path("r2bc_left_1ep_unfreeze_update_100.jsonl")

out_path = Path("gt_voxel_distance_freeze_vs_unfreeze.png")


# =========================
# load
# =========================
freeze_dist_df = mean_voxel_distance_from_log(freeze_log_path, "freeze")
unfreeze_dist_df = mean_voxel_distance_from_log(unfreeze_log_path, "unfreeze")

merged_dist = pd.merge(
    freeze_dist_df,
    unfreeze_dist_df,
    on="update",
    how="outer",
).sort_values("update")


# =========================
# plot
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    merged_dist["update"],
    merged_dist["freeze_gt_voxel_distance"],
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="Freeze",
    color="blue",
)

plt.plot(
    merged_dist["update"],
    merged_dist["unfreeze_gt_voxel_distance"],
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="Unfreeze",
    color="red",
)

plt.xlabel("Update", fontsize=20)
plt.ylabel("Mean GT voxel distance", fontsize=20)
plt.title("GT Voxel Distance: Freeze vs Unfreeze", fontsize=20)

plt.legend(fontsize=15)
plt.tick_params(axis="both", labelsize=15)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()


# =========================
# summary
# =========================
print(f"Saved: {out_path}")
print(f"freeze log:   {freeze_log_path}")
print(f"unfreeze log: {unfreeze_log_path}")

print(f"freeze points:   {freeze_dist_df['freeze_gt_voxel_distance'].notna().sum()}")
print(f"unfreeze points: {unfreeze_dist_df['unfreeze_gt_voxel_distance'].notna().sum()}")

print(
    f"freeze first:   {freeze_dist_df['freeze_gt_voxel_distance'].iloc[0]:.4f}, "
    f"last: {freeze_dist_df['freeze_gt_voxel_distance'].iloc[-1]:.4f}, "
    f"min: {freeze_dist_df['freeze_gt_voxel_distance'].min():.4f}"
)

print(
    f"unfreeze first: {unfreeze_dist_df['unfreeze_gt_voxel_distance'].iloc[0]:.4f}, "
    f"last: {unfreeze_dist_df['unfreeze_gt_voxel_distance'].iloc[-1]:.4f}, "
    f"min: {unfreeze_dist_df['unfreeze_gt_voxel_distance'].min():.4f}"
)