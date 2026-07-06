# Standard Library
import copy
import os
import sys
from os.path import dirname, abspath, join
from typing import Dict, Any

# Project paths
PROJECT_ROOT = dirname(abspath(__file__))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))

# Third Party
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Local
from agents.agent_factory import create_agent
from r2bc.datasets.r2bc_disk_buffer import (
    R2BCDiskDataset,
    r2bc_disk_collate_fn,
)
from r2bc.peract_update import (
    get_qagent,
    build_replay_sample,
    find_one_arm_batch,
    is_finite_loss,
    update_from_disk_buffer,
)


def print_tensor_shapes(title: str, data: Dict[str, Any]):
    print(f"\n[debug] {title}")
    for k, v in data.items():
        if torch.is_tensor(v):
            print(f"  {k}: Tensor shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")


def clone_named_params(module, only_trainable: bool = False):
    """
    module.named_parameters() から parameter を clone する。
    only_trainable=True の場合は requires_grad=True のものだけを見る。
    """
    cloned = {}

    for name, p in module.named_parameters():
        if only_trainable and not p.requires_grad:
            continue

        cloned[name] = p.detach().clone()

    return cloned


def summarize_param_diffs(before, after, title: str, max_print: int = 20):
    """
    before / after の parameter 差分を表示する。
    """
    changed = []
    unchanged = []
    missing = []

    for name, p_before in before.items():
        if name not in after:
            missing.append(name)
            continue

        p_after = after[name]
        diff = (p_after - p_before).abs().max().item()

        if diff > 0.0:
            changed.append((name, diff))
        else:
            unchanged.append(name)

    print(f"\n[debug] parameter diff summary: {title}")
    print(f"  num params checked: {len(before)}")
    print(f"  num changed: {len(changed)}")
    print(f"  num unchanged: {len(unchanged)}")
    print(f"  num missing: {len(missing)}")

    if changed:
        print("  changed params:")
        for name, diff in changed[:max_print]:
            print(f"    {name}: max_abs_diff={diff:.8e}")

    if missing:
        print("  missing params:")
        for name in missing[:max_print]:
            print(f"    {name}")

    return changed, unchanged, missing


def print_update_result(step_idx, batch, out):
    loss = out.get("total_loss", None)

    if torch.is_tensor(loss):
        loss_value = float(loss.detach().cpu())
        finite = is_finite_loss(loss)
    else:
        loss_value = None
        finite = False

    print(
        "[debug] update",
        f"update_idx={step_idx}",
        f"episode_index={int(batch['episode_index'][0])}",
        f"step_index={int(batch['step_index'][0])}",
        f"target_arm={batch['target_arm'][0]}",
        f"loss={loss_value}",
        f"finite={finite}",
    )

    return finite


def filter_params_by_keywords(params, keywords):
    """
    名前に keywords のどれかを含む parameter だけ抽出する。
    """
    return {
        name: value
        for name, value in params.items()
        if any(keyword in name for keyword in keywords)
    }


def filter_params_excluding_keywords(params, keywords):
    """
    名前に keywords を含まない parameter だけ抽出する。
    """
    return {
        name: value
        for name, value in params.items()
        if not any(keyword in name for keyword in keywords)
    }


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # -------------------------
    # config overrides
    # -------------------------
    cfg.method.name = "PERACT_BC"
    cfg.method.agent_type = "independent"
    cfg.method.robot_name = "bimanual"

    # debug では SE(3) augmentation を切る
    cfg.method.transform_augmentation.apply_se3 = False

    cfg.framework.anybimanual = True
    cfg.framework.checkpoint_name_prefix = "checkpoint"

    cfg.ddp.num_devices = 1
    cfg.replay.batch_size = 1

    # debug params
    target_arm = "left"  # "right" or "left"
    num_updates = 10
    data_root = join(PROJECT_ROOT, "data", "r2bc_peract")
    task_name = "bimanual_lift_long_block"
    checkpoint_dir = "/home/tappei-m/Project/AnyBimanual_checkpoints/PERACT_BC_leader_as_independent"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[debug] method:", cfg.method.name)
    print("[debug] agent_type:", cfg.method.agent_type)
    print("[debug] robot_name:", cfg.method.robot_name)
    print("[debug] anybimanual:", cfg.framework.anybimanual)
    print("[debug] transform_augmentation.apply_se3:", cfg.method.transform_augmentation.apply_se3)
    print("[debug] device:", device)
    print("[debug] target_arm:", target_arm)

    # -------------------------
    # CLIP 用 training=False agent
    # -------------------------
    print("[debug] creating clip/act agent...")
    clip_cfg = copy.deepcopy(cfg)
    clip_cfg.method.name = "PERACT_BC"
    clip_cfg.method.agent_type = "independent"
    clip_cfg.method.robot_name = "bimanual"
    clip_cfg.method.transform_augmentation.apply_se3 = False
    clip_cfg.framework.anybimanual = True
    clip_cfg.framework.checkpoint_name_prefix = "checkpoint"
    clip_cfg.ddp.num_devices = 1
    clip_cfg.replay.batch_size = 1

    clip_agent = create_agent(clip_cfg)
    clip_agent.build(training=False, device=device)
    clip_qagent = get_qagent(clip_agent, target_arm)

    assert hasattr(clip_qagent, "_clip_rn50"), "clip_qagent does not have _clip_rn50"
    print("[debug] found clip:", type(clip_qagent._clip_rn50))

    # -------------------------
    # update 用 training=True agent
    # -------------------------
    print("[debug] creating train agent...")
    train_cfg = copy.deepcopy(cfg)
    train_cfg.method.name = "PERACT_BC"
    train_cfg.method.agent_type = "independent"
    train_cfg.method.robot_name = "bimanual"
    train_cfg.method.transform_augmentation.apply_se3 = False
    train_cfg.framework.anybimanual = True
    train_cfg.framework.checkpoint_name_prefix = "checkpoint"
    train_cfg.ddp.num_devices = 1
    train_cfg.replay.batch_size = 1

    train_agent = create_agent(train_cfg)

    print("[debug] train_cfg.replay.batch_size:", train_cfg.replay.batch_size)
    print(
        "[debug] train_cfg transform_augmentation.apply_se3:",
        train_cfg.method.transform_augmentation.apply_se3,
    )

    train_agent.build(training=True, device=device)

    print("[debug] loading weights from:", checkpoint_dir)
    train_agent.load_weights(checkpoint_dir)
    print("[debug] load success!")

    train_qagent = get_qagent(train_agent, target_arm)
    train_qagent._q.train()

    # # -------------------------
    # # dataset
    # # -------------------------
    # print("[debug] PROJECT_ROOT:", PROJECT_ROOT)
    # print("[debug] cwd:", os.getcwd())
    # print("[debug] data_root:", data_root)
    # print("[debug] task_name:", task_name)
    # print("[debug] expected data dir:", join(data_root, task_name))
    # print("[debug] exists:", os.path.exists(join(data_root, task_name)))

    # print("[debug] loading R2BCDiskDataset...")
    # dataset = R2BCDiskDataset(
    #     data_root=data_root,
    #     task_name=task_name,
    #     max_episodes=None,
    # )
    # print("[debug] dataset summary:", dataset.summary())

    # loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=r2bc_disk_collate_fn,
    # )

    # batch = find_one_arm_batch(loader, target_arm=target_arm)
    # if batch is None:
    #     raise RuntimeError(f"No transition found for target_arm={target_arm}")

    # # -------------------------
    # # multiple update debug
    # # -------------------------
    # print(f"[debug] starting multiple {target_arm}-only updates...")
    # print("[debug] num_updates:", num_updates)
    # print("[debug] target_arm:", target_arm)

    # # parameter diff: before all updates
    # print("[debug] cloning params before all updates...")

    # all_before = clone_named_params(train_qagent._q, only_trainable=False)
    # trainable_before = clone_named_params(train_qagent._q, only_trainable=True)

    # ab_keywords = ["skill_manager", "visual_aligner"]

    # ab_before = filter_params_by_keywords(all_before, ab_keywords)
    # non_ab_before = filter_params_excluding_keywords(all_before, ab_keywords)

    # losses = []
    # num_success = 0
    # num_skipped = 0
    # num_failed = 0

    # for update_idx, batch in enumerate(loader):
    #     # target_arm only
    #     batch_arm = batch["target_arm"][0]
    #     if batch_arm != target_arm:
    #         num_skipped += 1
    #         continue

    #     if num_success >= num_updates:
    #         break

    #     print("\n" + "=" * 80)
    #     print(f"[debug] building replay_sample for update {num_success}")
    #     print("  target_arm:", batch["target_arm"])
    #     print("  target_action_9d:", batch["target_action_9d"].shape)
    #     print("  episode_path:", batch["episode_path"])
    #     print("  step_index:", batch["step_index"])

    #     try:
    #         replay_sample = build_replay_sample(
    #             batch=batch,
    #             cfg=train_cfg,
    #             train_qagent=train_qagent,
    #             clip_qagent=clip_qagent,
    #             target_arm=target_arm,
    #             device=device,
    #             debug=(num_success == 0),
    #         )

    #         if num_success == 0:
    #             print_tensor_shapes("replay_sample", replay_sample)
    #             print("[debug] replay_sample target_arm_id:", replay_sample["target_arm_id"])

    #         out = train_qagent.update(
    #             step=num_success,
    #             replay_sample=replay_sample,
    #         )

    #         finite = print_update_result(num_success, batch, out)

    #         if not finite:
    #             print("[warning] non-finite loss detected. stopping.")
    #             break

    #         loss_value = float(out["total_loss"].detach().cpu())
    #         losses.append(loss_value)
    #         num_success += 1

    #     except Exception as e:
    #         num_failed += 1
    #         print("[error] update failed:")
    #         print("  update_idx:", update_idx)
    #         print("  num_success:", num_success)
    #         print("  batch_arm:", batch_arm)
    #         print("  episode_path:", batch["episode_path"])
    #         print("  step_index:", batch["step_index"])
    #         raise e

    # print("\n" + "=" * 80)
    # print("[debug] multiple update summary")
    # print("  num_success:", num_success)
    # print("  num_skipped:", num_skipped)
    # print("  num_failed:", num_failed)
    # print("  losses:", losses)

    # if losses:
    #     print("  loss_first:", losses[0])
    #     print("  loss_last:", losses[-1])
    #     print("  loss_min:", min(losses))
    #     print("  loss_max:", max(losses))

    # -------------------------
    # dataset / disk buffer
    # -------------------------
    print("[debug] PROJECT_ROOT:", PROJECT_ROOT)
    print("[debug] cwd:", os.getcwd())
    print("[debug] data_root:", data_root)
    print("[debug] task_name:", task_name)
    print("[debug] expected data dir:", join(data_root, task_name))
    print("[debug] exists:", os.path.exists(join(data_root, task_name)))

    # -------------------------
    # update_from_disk_buffer debug
    # -------------------------
    print(f"[debug] starting update_from_disk_buffer target_arm={target_arm}")
    print("[debug] num_updates:", num_updates)

    # parameter diff: before all updates
    print("[debug] cloning params before all updates...")

    all_before = clone_named_params(train_qagent._q, only_trainable=False)
    trainable_before = clone_named_params(train_qagent._q, only_trainable=True)

    ab_keywords = ["skill_manager", "visual_aligner"]

    ab_before = filter_params_by_keywords(all_before, ab_keywords)
    non_ab_before = filter_params_excluding_keywords(all_before, ab_keywords)

    stats = update_from_disk_buffer(
        train_agent=train_agent,
        clip_agent=clip_agent,
        cfg=train_cfg,
        device=device,
        data_root=data_root,
        task_name=task_name,
        num_updates=num_updates,
        batch_size=1,
        max_episodes=None,
        target_arm=target_arm,
        shuffle=True,
        debug=True,
        raise_on_error=True,
    )

    print("\n" + "=" * 80)
    print("[debug] update_from_disk_buffer stats")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # -------------------------
    # parameter diff after all updates
    # -------------------------
    print("[debug] cloning params after all updates...")

    all_after = clone_named_params(train_qagent._q, only_trainable=False)
    trainable_after = clone_named_params(train_qagent._q, only_trainable=True)

    ab_after = filter_params_by_keywords(all_after, ab_keywords)
    non_ab_after = filter_params_excluding_keywords(all_after, ab_keywords)

    changed_trainable, unchanged_trainable, _ = summarize_param_diffs(
        trainable_before,
        trainable_after,
        title="trainable params only after multiple updates",
    )

    changed_ab, unchanged_ab, _ = summarize_param_diffs(
        ab_before,
        ab_after,
        title="AnyBimanual params after multiple updates",
    )

    changed_non_ab, unchanged_non_ab, _ = summarize_param_diffs(
        non_ab_before,
        non_ab_after,
        title="non-AnyBimanual params after multiple updates",
        max_print=10,
    )

    print("\n[debug] final check after update_from_disk_buffer:")
    print("  num_success:", stats["num_success"])
    print("  num_failed:", stats["num_failed"])
    print("  trainable params changed:", len(changed_trainable) > 0)
    print("  AnyBimanual params changed:", len(changed_ab) > 0)
    print("  non-AnyBimanual params changed:", len(changed_non_ab) > 0)

    if stats["num_success"] == 0:
        print("[warning] No updates were executed.")

    if len(changed_trainable) == 0:
        print("[warning] No trainable parameters changed. Optimizer may not be updating.")

    if len(changed_ab) == 0:
        print("[warning] SkillManager / VisualAligner parameters did not change.")

    if len(changed_non_ab) > 0:
        print("[warning] Some non-AnyBimanual parameters changed. Freeze may be incomplete.")


if __name__ == "__main__":
    main()