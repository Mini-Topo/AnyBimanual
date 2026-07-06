import argparse
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from hydra.experimental import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from agents.agent_factory import create_agent

from r2bc.datasets.r2bc_disk_buffer import R2BCDiskDataset
from r2bc.peract_update import (
    build_replay_sample,
    encode_lang_from_obs,
    get_qagent,
)


def tensor_summary(name, x):
    if x is None:
        print(f"[summary] {name}: None")
        return

    if isinstance(x, np.ndarray):
        arr = x
        arr_float = arr.astype(np.float32, copy=False)
        print(
            f"[summary] {name}: numpy "
            f"shape={arr.shape} dtype={arr.dtype} "
            f"min={np.nanmin(arr_float):.6f} max={np.nanmax(arr_float):.6f} "
            f"mean={np.nanmean(arr_float):.6f} "
            f"nan={np.isnan(arr_float).sum()} inf={np.isinf(arr_float).sum()}"
        )
        return

    if torch.is_tensor(x):
        t = x.detach()
        finite = torch.isfinite(t)
        if finite.any():
            t_float = t.float()
            print(
                f"[summary] {name}: torch "
                f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                f"min={float(t_float[finite].min()):.6f} "
                f"max={float(t_float[finite].max()):.6f} "
                f"mean={float(t_float[finite].mean()):.6f} "
                f"nan={int(torch.isnan(t_float).sum())} "
                f"inf={int(torch.isinf(t_float).sum())}"
            )
        else:
            print(
                f"[summary] {name}: torch "
                f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                f"all non-finite"
            )
        return

    print(f"[summary] {name}: type={type(x)} value={x}")


def print_replay_sample_summary(replay_sample):
    print("=" * 80)
    print("[replay_sample keys]")
    for k in sorted(replay_sample.keys()):
        v = replay_sample[k]
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        else:
            print(f"  {k}: type={type(v)}")

    print("=" * 80)
    print("[rgb / pcd / low_dim summaries]")
    for k in sorted(replay_sample.keys()):
        if (
            "rgb" in k
            or "point_cloud" in k
            or "low_dim_state" in k
            or "lang" in k
        ):
            tensor_summary(k, replay_sample[k])

    print("=" * 80)
    print("[action label summaries]")
    for k in [
        "trans_action_indicies",
        "rot_grip_action_indicies",
        "gripper_pose",
        "ignore_collisions",
    ]:
        if k in replay_sample:
            tensor_summary(k, replay_sample[k])
            print(f"[value] {k}: {replay_sample[k]}")

    print("=" * 80)


def load_cfg(config_dir, config_name="config"):
    """
    Hydra config を正しく読む。

    注意:
        OmegaConf.load(conf/config.yaml) だけだと defaults が展開されない。
        そのため conf/method/PERACT_BC.yaml が読み込まれず、
        cfg.method が None / 未定義になる。
    """
    config_dir = os.path.abspath(config_dir)
    cfg_path = Path(config_dir) / f"{config_name}.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        "method=PERACT_BC",
        "method.agent_type=independent",
        "framework.anybimanual=True",
        "framework.checkpoint_name_prefix=checkpoint",
        "ddp.num_devices=1",
        "rlbench.scene_bounds=[-0.6,-0.5,0.6,0.7,0.5,1.6]",
    ]

    print("[inspect] config_dir:", config_dir)
    print("[inspect] config_name:", config_name)
    print("[inspect] overrides:")
    for override in overrides:
        print("  ", override)

    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
        )

    print("[inspect] cfg loaded")
    print("[inspect] method:", cfg.method.name)
    print("[inspect] agent_type:", cfg.method.agent_type)
    print("[inspect] anybimanual:", cfg.framework.anybimanual)
    print("[inspect] checkpoint_name_prefix:", cfg.framework.checkpoint_name_prefix)
    print("[inspect] voxel_sizes:", cfg.method.voxel_sizes)
    print("[inspect] low_dim_size:", cfg.method.low_dim_size)

    return cfg


def create_clip_qagent(cfg, device):
    """
    encode_lang_from_obs() 用に qagent を作る。
    CLIP encoder (_clip_rn50) は training=False 側で作られる可能性が高い。
    """
    agent = create_agent(cfg)
    agent.build(training=False, device=device)

    qagent = get_qagent(agent, "right")

    print("[inspect] clip qagent type:", type(qagent))
    print("[inspect] has _clip_rn50:", hasattr(qagent, "_clip_rn50"))

    if not hasattr(qagent, "_clip_rn50"):
        print("[inspect] qagent attrs:", sorted(vars(qagent).keys()))
        raise AttributeError("clip qagent does not have _clip_rn50")

    return agent, qagent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="例: data/r2bc_keyframe_armfix_12ep_notrain",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="例: bimanual_lift_long_block",
    )
    parser.add_argument(
        "--target-arm",
        type=str,
        default=None,
        choices=[None, "right", "left", "both"],
        help="inspect対象。both/Noneなら両方。",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--config-dir",
        type=str,
        default="conf",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[inspect] device: {device}")

    target_arm = args.target_arm
    if target_arm == "both":
        target_arm = None

    print("[inspect] loading dataset...")
    dataset = R2BCDiskDataset(
        data_root=args.data_root,
        task_name=args.task,
    )

    print("[inspect] dataset summary:")
    if hasattr(dataset, "summary"):
        print(dataset.summary())
    else:
        print("  len:", len(dataset))

    print("[inspect] loading cfg...")
    cfg = load_cfg(args.config_dir)

    print("[inspect] creating clip qagent...")
    _, clip_qagent = create_clip_qagent(cfg, device)

    n = args.num_samples
    print(f"[inspect] num inspect samples: {n}")

    arm_counter = Counter()
    inspected = 0
    dataset_i = 0

    while inspected < n and dataset_i < len(dataset):
        sample = dataset[dataset_i]
        dataset_i += 1

        sample_arm = sample.get("target_arm", None)

        if target_arm is not None and sample_arm != target_arm:
            continue

        print("\n" + "#" * 100)
        print(f"[sample {inspected}] dataset_idx={dataset_i - 1}")

        arm_counter[sample_arm] += 1

        print("[raw dataset sample keys]")
        for k in sorted(sample.keys()):
            v = sample[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: np shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: type={type(v)} value={v if k not in ['obs', 'next_obs'] else '<dict>'}")

        arm = sample.get("target_arm", None)

        print("=" * 80)
        print("[raw transition]")
        print("  target_arm:", sample.get("target_arm"))
        print("  target_arm_id:", sample.get("target_arm_id"))
        print("  episode_index:", sample.get("episode_index"))
        print("  step_index:", sample.get("step_index"))

        if "target_action_9d" in sample:
            a = np.asarray(sample["target_action_9d"])
            print("  target_action_9d:", a)
            print("    xyz:", a[:3])
            print("    quat:", a[3:7])
            print("    gripper_open:", a[7])
            print("    ignore_collisions:", a[8])

        if "human_action" in sample:
            print("  human_action:", sample["human_action"])

        obs = sample["obs"]

        print("=" * 80)
        print("[raw obs summaries]")
        for k in sorted(obs.keys()):
            if (
                "rgb" in k
                or "point_cloud" in k
                or "low_dim_state" in k
                or "lang" in k
            ):
                tensor_summary(f"raw obs/{k}", obs[k])

        print("=" * 80)
        print("[encode lang]")
        lang_goal_emb, lang_token_embs = encode_lang_from_obs(
            obs=obs,
            clip_qagent=clip_qagent,
            device=device,
            debug=args.debug,
        )
        tensor_summary("lang_goal_emb", lang_goal_emb)
        tensor_summary("lang_token_embs", lang_token_embs)

        print("=" * 80)
        print("[build replay_sample]")

        batch = {
            "obs": [sample["obs"]],
            "target_action_9d": torch.as_tensor(
                sample["target_action_9d"][None],
                dtype=torch.float32,
                device=device,
            ),
            "target_arm": [arm],
        }

        replay_sample = build_replay_sample(
            batch=batch,
            cfg=cfg,
            train_qagent=clip_qagent,
            clip_qagent=clip_qagent,
            target_arm=arm,
            device=device,
            debug=args.debug,
        )

        print_replay_sample_summary(replay_sample)

        inspected += 1

    print("\n" + "=" * 80)
    print("[inspect done]")
    print("arm_counter:", dict(arm_counter))


if __name__ == "__main__":
    main()