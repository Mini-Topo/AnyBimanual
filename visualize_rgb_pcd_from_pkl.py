import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PKL_PATH = Path(
    "data/r2bc_eval_left_quat_current_grip_flip_debug/"
    "bimanual_lift_long_block/right_human/episode_000000.pkl"
)

STEP_ID = 0
CAMERA = "overhead"  # front / overhead / wrist_right / wrist_left など


def chw_to_hwc(x):
    """(3,H,W) -> (H,W,3)"""
    return np.transpose(x, (1, 2, 0))


def main():
    with open(PKL_PATH, "rb") as f:
        ep = pickle.load(f)

    step = ep["steps"][STEP_ID]
    obs = step["obs"]

    rgb = obs[f"{CAMERA}_rgb"]              # (3, H, W), uint8
    pcd = obs[f"{CAMERA}_point_cloud"]      # (3, H, W), float16/float32

    rgb_hwc = chw_to_hwc(rgb)
    pcd_hwc = chw_to_hwc(pcd).astype(np.float32)

    print("rgb shape:", rgb.shape, rgb.dtype)
    print("pcd shape:", pcd.shape, pcd.dtype)
    print("pcd min xyz:", np.nanmin(pcd_hwc.reshape(-1, 3), axis=0))
    print("pcd max xyz:", np.nanmax(pcd_hwc.reshape(-1, 3), axis=0))

    out_dir = Path("debug_vis_rgb_pcd")
    out_dir.mkdir(exist_ok=True)

    # 1. RGB画像
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_hwc)
    plt.title(f"{CAMERA} RGB")
    plt.axis("off")
    plt.tight_layout()
    rgb_path = out_dir / f"{CAMERA}_step{STEP_ID}_rgb.png"
    plt.savefig(rgb_path, dpi=150)
    plt.close()

    # 2. 点群の Z 座標を画像として表示
    z = pcd_hwc[:, :, 2]
    valid = np.isfinite(z)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.where(valid, z, np.nan), cmap="viridis")
    plt.title(f"{CAMERA} point cloud Z")
    plt.axis("off")
    plt.colorbar(label="Z")
    plt.tight_layout()
    z_path = out_dir / f"{CAMERA}_step{STEP_ID}_pcd_z.png"
    plt.savefig(z_path, dpi=150)
    plt.close()

    # 3. 点群を 3D scatter で表示
    xyz = pcd_hwc.reshape(-1, 3)
    colors = rgb_hwc.reshape(-1, 3).astype(np.float32) / 255.0

    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    colors = colors[valid]

    # 点数が多いので間引く
    max_points = 20000
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=1,
        alpha=0.8,
    )

    ax.set_title(f"{CAMERA} point cloud 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 見やすさ用：軸スケールをだいたい揃える
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    pcd_3d_path = out_dir / f"{CAMERA}_step{STEP_ID}_pcd_3d.png"
    plt.savefig(pcd_3d_path, dpi=150)
    plt.close()

    print("saved:")
    print(" ", rgb_path)
    print(" ", z_path)
    print(" ", pcd_3d_path)


if __name__ == "__main__":
    main()
