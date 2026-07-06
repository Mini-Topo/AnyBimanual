from typing import Dict, Any, Optional

import json
import os

import numpy as np
import torch
from omegaconf import DictConfig

from helpers import utils

from torch.utils.data import DataLoader, Subset

from r2bc.datasets.r2bc_disk_buffer import (
    R2BCDiskDataset,
    r2bc_disk_collate_fn,
)


CAMERAS = [
    "over_shoulder_right",
    "wrist_right",
    "front",
    "overhead",
    "over_shoulder_left",
    "wrist_left",
]

def r2bc_debug_enabled() -> bool:
    return os.environ.get("R2BC_VERBOSE_DEBUG", "0") == "1"


def r2bc_debug_print(*args, **kwargs):
    if r2bc_debug_enabled():
        print(*args, **kwargs)

def get_qagent(agent, arm: str):
    if arm == "right":
        return agent.right_agent._pose_agent._qattention_agents[0]
    if arm == "left":
        return agent.left_agent._pose_agent._qattention_agents[0]
    raise ValueError(f"Unknown arm: {arm}")


def arm_to_id(arm: str) -> int:
    if arm == "right":
        return 0
    if arm == "left":
        return 1
    raise ValueError(f"Unknown arm: {arm}")


def id_to_arm(arm_id: int) -> str:
    if arm_id == 0:
        return "right"
    if arm_id == 1:
        return "left"
    raise ValueError(f"Unknown arm_id: {arm_id}")


def encode_lang_from_obs(
    obs: Dict[str, Any],
    clip_qagent,
    device: torch.device,
    debug: bool = False,
):
    lang_goal_tokens = torch.as_tensor(
        obs["lang_goal_tokens"],
        dtype=torch.long,
        device=device,
    )

    # pkl: [77] -> [1, 77]
    if lang_goal_tokens.ndim == 1:
        lang_goal_tokens = lang_goal_tokens.unsqueeze(0)

    # 念のため [B, T, 77] -> [B, 77]
    if lang_goal_tokens.ndim == 3:
        lang_goal_tokens = lang_goal_tokens[:, -1]

    if debug:
        r2bc_debug_print("[peract_update] lang_goal_tokens shape:", tuple(lang_goal_tokens.shape))

    with torch.no_grad():
        lang_goal_emb, lang_token_embs = (
            clip_qagent._clip_rn50.encode_text_with_embeddings(
                lang_goal_tokens
            )
        )

    if lang_goal_emb.ndim == 1:
        lang_goal_emb = lang_goal_emb.unsqueeze(0)

    if lang_token_embs.ndim == 2:
        lang_token_embs = lang_token_embs.unsqueeze(0)

    if debug:
        r2bc_debug_print("[peract_update] lang_goal_emb shape:", tuple(lang_goal_emb.shape))
        r2bc_debug_print("[peract_update] lang_token_embs shape:", tuple(lang_token_embs.shape))

    return lang_goal_emb.float().detach(), lang_token_embs.float().detach()


def action_9d_to_peract_labels(
    action_9d,
    cfg: DictConfig,
    qagent,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    action_9d:
      [B, 9]
      [x, y, z, qx, qy, qz, qw, gripper_open, ignore_collisions]

    returns:
      trans_action_indicies:    [B, 3]
      rot_grip_action_indicies: [B, 4]
      gripper_pose:             [B, 7]
      ignore_collisions:        [B, 1]
    """
    if torch.is_tensor(action_9d):
        action_np = action_9d.detach().cpu().numpy()
    else:
        action_np = np.asarray(action_9d, dtype=np.float32)

    if action_np.ndim == 1:
        action_np = action_np[None]

    voxel_size = int(cfg.method.voxel_sizes[0])
    rotation_resolution = int(cfg.method.rotation_resolution)

    if hasattr(qagent, "_coordinate_bounds"):
        bounds = qagent._coordinate_bounds.detach().cpu().numpy()
        if bounds.ndim == 2:
            bounds = bounds[0]
    else:
        bounds = np.asarray(cfg.rlbench.scene_bounds, dtype=np.float32)

    trans_indices = []
    rot_grip_indices = []
    gripper_poses = []
    ignore_collisions = []

    for a in action_np:
        xyz = a[:3]
        quat = utils.normalize_quaternion(a[3:7])

        if quat[-1] < 0:
            quat = -quat

        disc_rot = utils.quaternion_to_discrete_euler(
            quat,
            rotation_resolution,
        )
        disc_rot = utils.correct_rotation_instability(
            disc_rot,
            rotation_resolution,
        )

        trans_idx = utils.point_to_voxel_index(
            xyz,
            voxel_size,
            bounds,
        )

        grip = int(a[7] > 0.5)
        ignore = int(a[8] > 0.5)

        trans_indices.append(np.asarray(trans_idx, dtype=np.int64))
        rot_grip_indices.append(
            np.asarray(
                [disc_rot[0], disc_rot[1], disc_rot[2], grip],
                dtype=np.int64,
            )
        )
        gripper_poses.append(a[:7].astype(np.float32))
        ignore_collisions.append(np.asarray([ignore], dtype=np.int64))

    return {
        "trans_action_indicies": torch.as_tensor(
            np.stack(trans_indices),
            dtype=torch.long,
            device=device,
        ),
        "rot_grip_action_indicies": torch.as_tensor(
            np.stack(rot_grip_indices),
            dtype=torch.long,
            device=device,
        ),
        "gripper_pose": torch.as_tensor(
            np.stack(gripper_poses),
            dtype=torch.float32,
            device=device,
        ),
        "ignore_collisions": torch.as_tensor(
            np.stack(ignore_collisions),
            dtype=torch.long,
            device=device,
        ),
    }


def obs_to_replay_inputs(
    obs: Dict[str, Any],
    arm: str,
    clip_qagent,
    device: torch.device,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    R2BC pkl の obs dict から PerAct update 用の observation 部分を作る。
    現状は batch_size=1 前提で、[C, H, W] -> [1, C, H, W] にする。
    """
    out: Dict[str, torch.Tensor] = {}

    for cam in CAMERAS:
        rgb_key = f"{cam}_rgb"
        pcd_key = f"{cam}_point_cloud"

        rgb = np.asarray(obs[rgb_key])
        pcd = np.asarray(obs[pcd_key])

        rgb_t = torch.as_tensor(
            rgb[None],
            dtype=torch.float32,
            device=device,
        )

        # Match PreprocessAgent._norm_rgb_ with norm_type='zero_mean'
        rgb_t = (rgb_t / 255.0) * 2.0 - 1.0

        out[rgb_key] = rgb_t
        
        out[pcd_key] = torch.as_tensor(
            pcd[None],
            dtype=torch.float32,
            device=device,
        )

    low_dim_key = f"{arm}_low_dim_state"
    low_dim = np.asarray(obs[low_dim_key], dtype=np.float32)

    out["low_dim_state"] = torch.as_tensor(
        low_dim[None],
        dtype=torch.float32,
        device=device,
    )

    lang_goal_emb, lang_token_embs = encode_lang_from_obs(
        obs=obs,
        clip_qagent=clip_qagent,
        device=device,
        debug=debug,
    )

    out["lang_goal_emb"] = lang_goal_emb.to(device)
    out["lang_token_embs"] = lang_token_embs.to(device)

    return out


def build_replay_sample(
    batch: Dict[str, Any],
    cfg: DictConfig,
    train_qagent,
    clip_qagent,
    target_arm: str,
    device: torch.device,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    R2BCDiskDataset の batch から PerAct update 用 replay_sample を作る。
    現状は batch_size=1 前提。
    """
    obs = batch["obs"][0]
    action_9d = batch["target_action_9d"]

    replay_sample: Dict[str, torch.Tensor] = {}

    replay_sample.update(
        action_9d_to_peract_labels(
            action_9d=action_9d,
            cfg=cfg,
            qagent=train_qagent,
            device=device,
        )
    )

    replay_sample.update(
        obs_to_replay_inputs(
            obs=obs,
            arm=target_arm,
            clip_qagent=clip_qagent,
            device=device,
            debug=debug,
        )
    )

    replay_sample["target_arm_id"] = torch.tensor(
        [arm_to_id(target_arm)],
        dtype=torch.long,
        device=device,
    )

    return replay_sample

def _slice_batch(batch: Dict[str, Any], i: int) -> Dict[str, Any]:
    """
    r2bc_disk_collate_fn が返した batch から i 番目だけを取り出し、
    batch_size=1 の形に戻す。
    """
    out = {}

    for k, v in batch.items():
        if k in ("obs", "next_obs", "meta", "info", "episode_path", "target_arm"):
            # list系: [item0, item1, ...] -> [item_i]
            out[k] = [v[i]]
        elif torch.is_tensor(v):
            # Tensor系: [B, ...] -> [1, ...]
            out[k] = v[i:i + 1]
        elif isinstance(v, np.ndarray):
            out[k] = v[i:i + 1]
        elif isinstance(v, list):
            out[k] = [v[i]]
        else:
            # 念のため
            out[k] = v

    return out


def _concat_replay_samples(samples):
    """
    build_replay_sample() が返した batch_size=1 の replay_sample 群を
    batch_size=B の replay_sample にまとめる。
    """
    if len(samples) == 0:
        raise ValueError("Cannot concatenate empty replay sample list.")

    out = {}

    for k in samples[0].keys():
        vals = [s[k] for s in samples]

        if torch.is_tensor(vals[0]):
            if vals[0].dim() == 0:
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = torch.cat(vals, dim=0)
        else:
            out[k] = vals

    return out


def _as_list(x):
    if isinstance(x, list):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]

def find_one_arm_batch(loader, target_arm: str) -> Optional[Dict[str, Any]]:
    for batch in loader:
        if batch["target_arm"][0] == target_arm:
            return batch
    return None


def is_finite_loss(loss) -> bool:
    if not torch.is_tensor(loss):
        return False
    return bool(torch.isfinite(loss.detach()).all().item())


def update_from_batch(
    train_agent,
    clip_agent,
    batch: Dict[str, Any],
    cfg: DictConfig,
    device: torch.device,
    step: int,
    target_arm: Optional[str] = None,
    debug: bool = False,
    zero_grad: bool = True,
    step_optimizer: bool = True,
    loss_scale: float = 1.0,
):
    """
    1 batch 分だけ PerAct/AnyBimanual update する。
    batch_size > 1 は target_arm が "right" or "left" のときのみ対応。
    """
    batch_arms = batch["target_arm"]

    if target_arm is None:
        target_arm = batch_arms[0]

    # batch 内に異なる arm が混ざるのはまだ禁止
    for arm in batch_arms:
        if arm != target_arm:
            raise NotImplementedError(
                f"Mixed-arm batch is not supported yet. target_arm={target_arm}, got arm={arm}"
            )

    train_qagent = get_qagent(train_agent, target_arm)
    clip_qagent = get_qagent(clip_agent, target_arm)

    batch_size_actual = len(batch_arms)
    replay_samples = []

    for i in range(batch_size_actual):
        single_batch = _slice_batch(batch, i)

        replay_sample_i = build_replay_sample(
            batch=single_batch,
            cfg=cfg,
            train_qagent=train_qagent,
            clip_qagent=clip_qagent,
            target_arm=target_arm,
            device=device,
            debug=debug and i == 0,
        )
        replay_samples.append(replay_sample_i)

    replay_sample = _concat_replay_samples(replay_samples)

    before_params = snapshot_trainable_params(train_qagent)

    skill_hook_handle = None
    if debug:
        skill_hook_handle = attach_skill_manager_debug_hook(train_qagent)

    try:
        out = train_qagent.update(
            step=step,
            replay_sample=replay_sample,
            zero_grad=zero_grad,
            step_optimizer=step_optimizer,
            loss_scale=loss_scale,
        )
    finally:
        if skill_hook_handle is not None:
            skill_hook_handle.remove()

    out["module_grad_stats"] = summarize_grads_by_module(train_qagent)
    out["module_param_delta_stats"] = summarize_param_delta_by_module(
        train_qagent,
        before_params,
    )

    # Attach detailed losses recorded inside QAttentionPerActBCAgent.update().
    if hasattr(train_qagent, "_summaries"):
        for k, v in train_qagent._summaries.items():
            out[k] = v

    # Attach prediction / GT voxel coordinates for quick sanity checking.
    for attr_name, out_key in [
        ("_vis_gt_coordinate", "vis_gt_coordinate"),
        ("_vis_max_coordinate", "vis_max_coordinate"),
    ]:
        if hasattr(train_qagent, attr_name):
            out[out_key] = getattr(train_qagent, attr_name)

    return out, replay_sample

def get_qnet_from_qagent(qagent):
    """
    qagent._q._qnet を取り出す。
    DDP の場合は .module まで剥がす。
    """
    qnet = qagent._q._qnet

    if hasattr(qnet, "module"):
        qnet = qnet.module

    return qnet

def _module_group_from_name(name: str) -> str:
    if "skill_manager" in name:
        return "skill_manager"
    if "visual_aligner" in name:
        return "visual_aligner"
    if "trans_decoder" in name:
        return "trans_decoder"
    if "rot_grip_collision_ff" in name:
        return "rot_grip_collision_ff"
    if "dense0" in name or "dense1" in name:
        return "rot_grip_dense"
    if "final.conv3d" in name:
        return "final_conv3d"
    return "peract_core"


def summarize_trainable_modules(qagent):
    """
    qagent 内の module ごとの trainable / optimizer 登録状況を見る。
    単腕 PerAct 本体が固定されているか確認するための関数。
    """
    qnet = get_qnet_from_qagent(qagent)

    optimizer_param_ids = set()
    if hasattr(qagent, "_optimizer") and qagent._optimizer is not None:
        for group in qagent._optimizer.param_groups:
            for p in group["params"]:
                optimizer_param_ids.add(id(p))

    stats = {}

    for name, p in qnet.named_parameters():
        group = _module_group_from_name(name)

        if group not in stats:
            stats[group] = {
                "num_params": 0,
                "num_tensors": 0,
                "requires_grad_params": 0,
                "requires_grad_tensors": 0,
                "optimizer_params": 0,
                "optimizer_tensors": 0,
                "examples": [],
            }

        s = stats[group]
        s["num_params"] += p.numel()
        s["num_tensors"] += 1

        if p.requires_grad:
            s["requires_grad_params"] += p.numel()
            s["requires_grad_tensors"] += 1

        if id(p) in optimizer_param_ids:
            s["optimizer_params"] += p.numel()
            s["optimizer_tensors"] += 1

        if len(s["examples"]) < 5:
            s["examples"].append(
                {
                    "name": name,
                    "shape": tuple(p.shape),
                    "requires_grad": bool(p.requires_grad),
                    "in_optimizer": id(p) in optimizer_param_ids,
                }
            )

    return stats


def print_trainable_module_summary(qagent, prefix="[module summary]"):
    stats = summarize_trainable_modules(qagent)

    print(prefix)
    for group, s in stats.items():
        r2bc_debug_print(
            f"  {group}: "
            f"num_params={s['num_params']} "
            f"requires_grad_params={s['requires_grad_params']} "
            f"optimizer_params={s['optimizer_params']} "
            f"num_tensors={s['num_tensors']} "
            f"requires_grad_tensors={s['requires_grad_tensors']} "
            f"optimizer_tensors={s['optimizer_tensors']}"
        )
        for ex in s["examples"]:
            r2bc_debug_print(
                f"    - {ex['name']} "
                f"shape={ex['shape']} "
                f"requires_grad={ex['requires_grad']} "
                f"in_optimizer={ex['in_optimizer']}"
            )


def snapshot_trainable_params(qagent):
    """
    trainable params のみ CPU に clone する。
    param_delta 計測用。
    """
    qnet = get_qnet_from_qagent(qagent)
    snap = {}

    for name, p in qnet.named_parameters():
        if p.requires_grad:
            snap[name] = p.detach().cpu().clone()

    return snap


def summarize_grads_by_module(qagent):
    """
    backward 後の grad norm を module ごとに集計。
    """
    qnet = get_qnet_from_qagent(qagent)

    stats = {}

    for name, p in qnet.named_parameters():
        group = _module_group_from_name(name)

        if group not in stats:
            stats[group] = {
                "num_tensors_with_grad": 0,
                "num_params_with_grad": 0,
                "grad_l2_sq": 0.0,
                "grad_abs_max": 0.0,
            }

        if p.grad is None:
            continue

        g = p.grad.detach()
        s = stats[group]
        s["num_tensors_with_grad"] += 1
        s["num_params_with_grad"] += p.numel()
        s["grad_l2_sq"] += float(torch.sum(g.float() ** 2).detach().cpu())
        s["grad_abs_max"] = max(
            s["grad_abs_max"],
            float(g.abs().max().detach().cpu()),
        )

    for group, s in stats.items():
        s["grad_l2"] = float(np.sqrt(s["grad_l2_sq"]))

    return stats


def summarize_param_delta_by_module(qagent, before_snapshot):
    """
    optimizer.step 前後の parameter delta を module ごとに集計。
    before_snapshot は snapshot_trainable_params() の返り値。
    """
    qnet = get_qnet_from_qagent(qagent)

    stats = {}

    for name, p in qnet.named_parameters():
        if name not in before_snapshot:
            continue

        group = _module_group_from_name(name)

        if group not in stats:
            stats[group] = {
                "num_changed_tensors": 0,
                "num_tracked_tensors": 0,
                "delta_l2_sq": 0.0,
                "delta_abs_max": 0.0,
            }

        before = before_snapshot[name].to(device=p.device)
        after = p.detach()
        d = after - before

        s = stats[group]
        s["num_tracked_tensors"] += 1

        delta_abs_max = float(d.abs().max().detach().cpu())
        if delta_abs_max > 0:
            s["num_changed_tensors"] += 1

        s["delta_l2_sq"] += float(torch.sum(d.float() ** 2).detach().cpu())
        s["delta_abs_max"] = max(s["delta_abs_max"], delta_abs_max)

    for group, s in stats.items():
        s["delta_l2"] = float(np.sqrt(s["delta_l2_sq"]))

    return stats

def sync_qagent_anybimanual_modules(src_qagent, dst_qagent):
    """
    src_qagent の SkillManager / VisualAligner を dst_qagent にコピーする。

    想定:
      src_qagent: training=True 側
      dst_qagent: 推論/収集に使う agent 側
    """
    src_qnet = get_qnet_from_qagent(src_qagent)
    dst_qnet = get_qnet_from_qagent(dst_qagent)

    if not hasattr(src_qnet, "skill_manager"):
        raise AttributeError("src_qnet does not have skill_manager")
    if not hasattr(src_qnet, "visual_aligner"):
        raise AttributeError("src_qnet does not have visual_aligner")
    if not hasattr(dst_qnet, "skill_manager"):
        raise AttributeError("dst_qnet does not have skill_manager")
    if not hasattr(dst_qnet, "visual_aligner"):
        raise AttributeError("dst_qnet does not have visual_aligner")

    dst_qnet.skill_manager.load_state_dict(
        src_qnet.skill_manager.state_dict()
    )
    dst_qnet.visual_aligner.load_state_dict(
        src_qnet.visual_aligner.state_dict()
    )


def sync_anybimanual_modules(src_agent, dst_agent, arms=("right", "left")):
    """
    src_agent の AnyBimanual modules を dst_agent に同期する。
    right / left の両方について SkillManager / VisualAligner をコピーする。
    """
    for arm in arms:
        src_qagent = get_qagent(src_agent, arm)
        dst_qagent = get_qagent(dst_agent, arm)

        sync_qagent_anybimanual_modules(
            src_qagent=src_qagent,
            dst_qagent=dst_qagent,
        )

def attach_skill_manager_debug_hook(qagent):
    qnet = get_qnet_from_qagent(qagent)

    if not hasattr(qnet, "skill_manager"):
        r2bc_debug_print("[skill debug] qnet has no skill_manager")
        return None

    def _summarize_tensor(x):
        if not torch.is_tensor(x):
            return None
        xd = x.detach()
        return {
            "shape": tuple(xd.shape),
            "mean": float(xd.float().mean().cpu()),
            "std": float(xd.float().std().cpu()),
            "abs_mean": float(xd.float().abs().mean().cpu()),
            "abs_max": float(xd.float().abs().max().cpu()),
            "requires_grad": bool(x.requires_grad),
        }

    def hook(module, inputs, output):
        r2bc_debug_print("[skill debug] SkillManager forward")

        r2bc_debug_print("  inputs:")
        for i, x in enumerate(inputs):
            if torch.is_tensor(x):
                r2bc_debug_print(f"    input[{i}]: {_summarize_tensor(x)}")
            elif isinstance(x, (list, tuple)):
                r2bc_debug_print(f"    input[{i}]: {type(x)} len={len(x)}")
                for j, y in enumerate(x[:3]):
                    if torch.is_tensor(y):
                        r2bc_debug_print(f"      [{j}]: {_summarize_tensor(y)}")
            else:
                r2bc_debug_print(f"    input[{i}]: type={type(x)}")

        r2bc_debug_print("  output:")
        if torch.is_tensor(output):
            r2bc_debug_print("   ", _summarize_tensor(output))
        elif isinstance(output, (list, tuple)):
            r2bc_debug_print(f"    type={type(output)} len={len(output)}")
            for i, y in enumerate(output):
                if torch.is_tensor(y):
                    r2bc_debug_print(f"      output[{i}]: {_summarize_tensor(y)}")
                else:
                    r2bc_debug_print(f"      output[{i}]: type={type(y)}")
        elif isinstance(output, dict):
            r2bc_debug_print("    dict keys:", list(output.keys()))
            for k, y in output.items():
                if torch.is_tensor(y):
                    r2bc_debug_print(f"      {k}: {_summarize_tensor(y)}")
                else:
                    r2bc_debug_print(f"      {k}: type={type(y)}")
        else:
            r2bc_debug_print("    type:", type(output))

    return qnet.skill_manager.register_forward_hook(hook)

def _print_module_stats_block(title, accum_outs):
    r2bc_debug_print(title)
    if not accum_outs:
        r2bc_debug_print("  None")
        return

    for i, x in enumerate(accum_outs):
        r2bc_debug_print(
            f"  accum_i={i} "
            f"step_index={x.get('step_index')} "
            f"gt={x.get('gt_voxel')} "
            f"pred={x.get('pred_voxel')}"
        )

        r2bc_debug_print("    grad:")
        for group, s in x.get("module_grad_stats", {}).items():
            r2bc_debug_print(
                f"      {group}: "
                f"grad_l2={s.get('grad_l2')} "
                f"grad_abs_max={s.get('grad_abs_max')}"
            )

        r2bc_debug_print("    delta:")
        for group, s in x.get("module_param_delta_stats", {}).items():
            r2bc_debug_print(
                f"      {group}: "
                f"delta_l2={s.get('delta_l2')} "
                f"delta_abs_max={s.get('delta_abs_max')} "
                f"num_changed_tensors={s.get('num_changed_tensors')}"
            )

def update_from_disk_buffer(
    train_agent,
    clip_agent,
    cfg: DictConfig,
    device: torch.device,
    data_root: str,
    task_name: str,
    num_updates: int,
    batch_size: int = 1,
    max_episodes: Optional[int] = None,
    target_arm: Optional[str] = None,
    shuffle: bool = True,
    debug: bool = False,
    raise_on_error: bool = True,
    grad_accum_steps: int = 1,
    progress_log_path: Optional[str] = None,
    progress_log_every: int = 100,
    save_ckpt_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    R2BCDiskDataset から transition を読み、PerAct/AnyBimanual を update する。

    現状の build_replay_sample() は batch_size=1 前提。
    そのため、ここでも batch_size=1 のみ許可する。

    Args:
        train_agent:
            training=True で build された BimanualAgent
        clip_agent:
            training=False で build された BimanualAgent
        cfg:
            train_agent と同じ Hydra config
        device:
            torch.device
        data_root:
            data/r2bc_peract など
        task_name:
            bimanual_lift_long_block など
        num_updates:
            実行する update 回数
        batch_size:
            現状は 1 のみ対応
        max_episodes:
            読み込む episode 数の上限
        target_arm:
            None なら right/left mixed update
            "right" なら right のみ
            "left" なら left のみ
        shuffle:
            DataLoader shuffle
        debug:
            初回 replay_sample shape などを出す
        raise_on_error:
            True なら update 失敗時に例外を投げる
            False なら失敗を stats に記録して続行を試みる

    Returns:
        stats dict
    """
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1. Got {grad_accum_steps}")

    if batch_size > 1 and target_arm is None:
        raise NotImplementedError(
            "batch_size > 1 is currently supported only when target_arm is 'right' or 'left'. "
            "Mixed right/left batches are not supported yet."
        )

    if target_arm not in (None, "right", "left"):
        raise ValueError(f"target_arm must be None, 'right', or 'left'. Got {target_arm}")

    dataset_all = R2BCDiskDataset(
        data_root=data_root,
        task_name=task_name,
        max_episodes=max_episodes,
    )

    dataset = dataset_all
    filtered_num_transitions = len(dataset_all)

    if target_arm in ("right", "left"):
        arm_indices = []

        for i in range(len(dataset_all)):
            sample = dataset_all[i]
            sample_arm = sample["target_arm"]

            if sample_arm == target_arm:
                arm_indices.append(i)

        if len(arm_indices) == 0:
            raise RuntimeError(
                f"No samples found for target_arm={target_arm}. "
                f"num_transitions={len(dataset_all)}, "
                f"dataset_summary={dataset_all.summary()}"
            )

        print(
            "[peract_update] filtered dataset: "
            f"target_arm={target_arm}, "
            f"{len(arm_indices)}/{len(dataset_all)} transitions"
        )

        dataset = Subset(dataset_all, arm_indices)
        filtered_num_transitions = len(arm_indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=r2bc_disk_collate_fn,
    )

    stats: Dict[str, Any] = {
        "num_success": 0,
        "num_skipped": 0,
        "num_failed": 0,
        "losses": [],
        "trans_losses": [],
        "rot_losses": [],
        "grip_losses": [],
        "collision_losses": [],
        "gt_voxels": [],
        "pred_voxels": [],
        "arms": [],
        "episode_indices": [],
        "step_indices": [],
        "target_arm": target_arm,
        "num_updates_requested": num_updates,
        "batch_size": batch_size,
        "dataset_summary": dataset_all.summary(),
        "filtered_num_transitions": filtered_num_transitions,
        "filtered_target_arm": target_arm,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
    }

    if debug:
        r2bc_debug_print("[peract_update] update_from_disk_buffer")
        r2bc_debug_print("[peract_update] data_root:", data_root)
        r2bc_debug_print("[peract_update] task_name:", task_name)
        r2bc_debug_print("[peract_update] num_updates:", num_updates)
        r2bc_debug_print("[peract_update] batch_size:", batch_size)
        r2bc_debug_print("[peract_update] target_arm:", target_arm)
        r2bc_debug_print("[peract_update] dataset summary:", dataset_all.summary())
        r2bc_debug_print("[peract_update] filtered_num_transitions:", filtered_num_transitions)
        r2bc_debug_print("[peract_update] grad_accum_steps:", grad_accum_steps)
        r2bc_debug_print("[peract_update] effective_batch_size:", batch_size * grad_accum_steps)

    if debug and target_arm in ("right", "left"):
        train_qagent = get_qagent(train_agent, target_arm)
        r2bc_debug_print_trainable_module_summary(
            train_qagent,
            prefix=f"[peract_update] trainable module summary arm={target_arm}",
        )

    epoch = 0
    loader_iter = iter(loader)

    def _to_float_or_none(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return float(x.detach().cpu())
        try:
            return float(x)
        except Exception:
            return None

    def _to_list_or_none(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        try:
            return list(x)
        except Exception:
            return None

    def _mean_non_none(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _json_safe(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_safe(v) for v in x]
        return x

    def _write_progress(record: Dict[str, Any]):
        if progress_log_path is None:
            return

        os.makedirs(os.path.dirname(progress_log_path), exist_ok=True)

        with open(progress_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def _next_matching_batch():
        """
        target_arm に合う batch を1つ返す。
        loader を1周しても見つからなければ (None, None) を返す。
        """
        nonlocal epoch, loader_iter

        checked = 0
        max_check = max(1, len(loader))

        while checked < max_check:
            try:
                batch = next(loader_iter)
            except StopIteration:
                epoch += 1
                loader_iter = iter(loader)
                continue

            checked += 1
            batch_arm = batch["target_arm"][0]

            if target_arm is not None and batch_arm != target_arm:
                stats["num_skipped"] += 1
                continue

            return batch, batch_arm

        return None, None

    while stats["num_success"] < num_updates:
        step = stats["num_success"]
        accum_outs = []

        for accum_i in range(grad_accum_steps):
            batch, batch_arm = _next_matching_batch()

            if batch is None:
                r2bc_debug_print(
                    "[peract_update] no matching target_arm found in dataset; "
                    f"target_arm={target_arm}. stopping."
                )
                break

            try:
                out, replay_sample = update_from_batch(
                    train_agent=train_agent,
                    clip_agent=clip_agent,
                    batch=batch,
                    cfg=cfg,
                    device=device,
                    step=step,
                    target_arm=batch_arm,
                    debug=debug and (step == 0 or step == num_updates - 1),
                    zero_grad=(accum_i == 0),
                    step_optimizer=(accum_i == grad_accum_steps - 1),
                    loss_scale=1.0 / grad_accum_steps,
                )

                should_print_step = debug and (
                    step == 0 or step == num_updates - 1 or step % 100 == 0
                )

                if should_print_step:
                    r2bc_debug_print("[peract_update] module grad stats:")
                    for group, s in out.get("module_grad_stats", {}).items():
                        r2bc_debug_print(
                            f"  {group}: "
                            f"grad_l2={s.get('grad_l2')} "
                            f"grad_abs_max={s.get('grad_abs_max')} "
                            f"num_tensors_with_grad={s.get('num_tensors_with_grad')} "
                            f"num_params_with_grad={s.get('num_params_with_grad')}"
                        )

                    r2bc_debug_print("[peract_update] module param delta stats:")
                    for group, s in out.get("module_param_delta_stats", {}).items():
                        r2bc_debug_print(
                            f"  {group}: "
                            f"delta_l2={s.get('delta_l2')} "
                            f"delta_abs_max={s.get('delta_abs_max')} "
                            f"num_changed_tensors={s.get('num_changed_tensors')} "
                            f"num_tracked_tensors={s.get('num_tracked_tensors')}"
                        )

                loss = out.get("total_loss", None)
                finite = is_finite_loss(loss)

                loss_value = _to_float_or_none(loss)
                trans_loss_value = _to_float_or_none(out.get("losses/trans_loss", None))
                rot_loss_value = _to_float_or_none(out.get("losses/rot_loss", None))
                grip_loss_value = _to_float_or_none(out.get("losses/grip_loss", None))
                collision_loss_value = _to_float_or_none(out.get("losses/collision_loss", None))
                gt_voxel = _to_list_or_none(out.get("vis_gt_coordinate", None))
                pred_voxel = _to_list_or_none(out.get("vis_max_coordinate", None))

                gt_rot_grip = _to_list_or_none(out.get("debug/gt_rot_grip", None))
                pred_rot_grip_raw = _to_list_or_none(out.get("debug/pred_rot_grip_raw", None))
                pred_rot_grip_act_style = _to_list_or_none(
                    out.get("debug/pred_rot_grip_act_style", None)
                )
                gt_grip = _to_float_or_none(out.get("debug/gt_grip", None))
                pred_grip_raw = _to_float_or_none(out.get("debug/pred_grip_raw", None))
                pred_grip_act_style = _to_float_or_none(
                    out.get("debug/pred_grip_act_style", None)
                )

                if should_print_step:
                    r2bc_debug_print(
                        "[peract_update] accum",
                        f"epoch={epoch}",
                        f"update_idx={step}",
                        f"accum_i={accum_i}/{grad_accum_steps}",
                        f"target_arm={batch_arm}",
                        f"episode_index={int(batch['episode_index'][0])}",
                        f"step_index={int(batch['step_index'][0])}",
                        f"zero_grad={accum_i == 0}",
                        f"step_optimizer={accum_i == grad_accum_steps - 1}",
                        f"loss_scale={1.0 / grad_accum_steps}",
                        f"loss={loss_value}",
                        f"trans={trans_loss_value}",
                        f"rot={rot_loss_value}",
                        f"grip={grip_loss_value}",
                        f"collision={collision_loss_value}",
                        f"gt={gt_voxel}",
                        f"pred={pred_voxel}",
                        f"finite={finite}",
                        f"gt_rot_grip={gt_rot_grip}",
                        f"pred_rot_grip_raw={pred_rot_grip_raw}",
                        f"pred_rot_grip_act_style={pred_rot_grip_act_style}",
                        f"gt_grip={gt_grip}",
                        f"pred_grip_raw={pred_grip_raw}",
                        f"pred_grip_act_style={pred_grip_act_style}",
                    )

                if not finite:
                    stats["num_failed"] += 1
                    error_msg = "non-finite loss detected"

                    if raise_on_error:
                        raise RuntimeError(error_msg)

                    stats.setdefault("errors", []).append(
                        {
                            "epoch": epoch,
                            "update_idx": step,
                            "accum_i": accum_i,
                            "target_arm": batch_arm,
                            "episode_index": int(batch["episode_index"][0]),
                            "step_index": int(batch["step_index"][0]),
                            "error": error_msg,
                        }
                    )
                    break

                accum_outs.append(
                    {
                        "loss": loss_value,
                        "trans_loss": trans_loss_value,
                        "rot_loss": rot_loss_value,
                        "grip_loss": grip_loss_value,
                        "collision_loss": collision_loss_value,
                        "gt_voxel": gt_voxel,
                        "pred_voxel": pred_voxel,
                        "arm": batch_arm,
                        "episode_index": int(batch["episode_index"][0]),
                        "step_index": int(batch["step_index"][0]),
                        "module_grad_stats": out.get("module_grad_stats", {}),
                        "module_param_delta_stats": out.get("module_param_delta_stats", {}),
                        "gt_rot_grip": gt_rot_grip,
                        "pred_rot_grip_raw": pred_rot_grip_raw,
                        "pred_rot_grip_act_style": pred_rot_grip_act_style,
                        "gt_grip": gt_grip,
                        "pred_grip_raw": pred_grip_raw,
                        "pred_grip_act_style": pred_grip_act_style,
                    }
                )

            except Exception as e:
                stats["num_failed"] += 1

                error_info = {
                    "epoch": epoch,
                    "update_idx": step,
                    "accum_i": accum_i,
                    "target_arm": batch_arm,
                    "episode_index": int(batch["episode_index"][0]),
                    "step_index": int(batch["step_index"][0]),
                    "error": repr(e),
                }
                stats.setdefault("errors", []).append(error_info)

                if debug:
                    r2bc_debug_print("[peract_update] update failed:")
                    for k, v in error_info.items():
                        r2bc_debug_print(f"  {k}: {v}")

                if raise_on_error:
                    raise

                break

        if len(accum_outs) != grad_accum_steps:
            r2bc_debug_print(
                "[peract_update] incomplete gradient accumulation; "
                f"got {len(accum_outs)} / {grad_accum_steps}. stopping."
            )
            break

        # ここに来た時点で、最後の accum_i で optimizer.step() 済み
        stats["losses"].append(_mean_non_none([x["loss"] for x in accum_outs]))
        stats["trans_losses"].append(_mean_non_none([x["trans_loss"] for x in accum_outs]))
        stats["rot_losses"].append(_mean_non_none([x["rot_loss"] for x in accum_outs]))
        stats["grip_losses"].append(_mean_non_none([x["grip_loss"] for x in accum_outs]))
        stats["collision_losses"].append(
            _mean_non_none([x["collision_loss"] for x in accum_outs])
        )

        # 代表値として最後の sample の voxel を記録
        stats["gt_voxels"].append(accum_outs[-1]["gt_voxel"])
        stats["pred_voxels"].append(accum_outs[-1]["pred_voxel"])
        stats["arms"].append(accum_outs[-1]["arm"])
        stats["episode_indices"].append(accum_outs[-1]["episode_index"])
        stats["step_indices"].append(accum_outs[-1]["step_index"])

        # 詳細確認用
        if debug and (step == 0 or step == num_updates - 1):
            stats.setdefault("accum_details", []).append(accum_outs)

        if stats["num_success"] == 0:
            stats["module_stats_first"] = accum_outs

        stats["module_stats_last"] = accum_outs


        next_success = stats["num_success"] + 1

        should_log_progress = (
            next_success == 1
            or next_success == num_updates
            or next_success % progress_log_every == 0
        )

        if should_log_progress:
            progress_record = {
                "update": next_success,
                "num_updates": num_updates,
                "target_arm": target_arm,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps,

                "loss": stats["losses"][-1],
                "trans_loss": stats["trans_losses"][-1],
                "rot_loss": stats["rot_losses"][-1],
                "grip_loss": stats["grip_losses"][-1],
                "collision_loss": stats["collision_losses"][-1],

                "loss_first": stats["losses"][0],
                "loss_delta": stats["losses"][-1] - stats["losses"][0],

                "last_step_index": stats["step_indices"][-1],
                "last_gt_voxel": stats["gt_voxels"][-1],
                "last_pred_voxel": stats["pred_voxels"][-1],

                "module_stats_last": stats.get("module_stats_last", []),
            }

            r2bc_debug_print(
                "[peract_update] progress "
                f"{next_success}/{num_updates} "
                f"loss={progress_record['loss']:.3f} "
                f"delta={progress_record['loss_delta']:.3f} "
                f"gt={progress_record['last_gt_voxel']} "
                f"pred={progress_record['last_pred_voxel']}"
            )

            _write_progress(progress_record)

        # num_success は optimizer step 数として数える
        stats["num_success"] += 1

    stats["num_epochs"] = epoch

    if stats["losses"]:
        stats["loss_first"] = stats["losses"][0]
        stats["loss_last"] = stats["losses"][-1]
        stats["loss_min"] = min(stats["losses"])
        stats["loss_max"] = max(stats["losses"])

        for key in ["trans_losses", "rot_losses", "grip_losses", "collision_losses"]:
            vals = [v for v in stats[key] if v is not None]
            if vals:
                stats[f"{key}_first"] = vals[0]
                stats[f"{key}_last"] = vals[-1]
                stats[f"{key}_min"] = min(vals)
                stats[f"{key}_max"] = max(vals)
            else:
                stats[f"{key}_first"] = None
                stats[f"{key}_last"] = None
                stats[f"{key}_min"] = None
                stats[f"{key}_max"] = None
    else:
        stats["loss_first"] = None
        stats["loss_last"] = None
        stats["loss_min"] = None
        stats["loss_max"] = None

    if save_ckpt_dir is not None:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        train_agent.save_weights(save_ckpt_dir)
        stats["saved_ckpt_dir"] = save_ckpt_dir
        r2bc_debug_print(f"[peract_update] saved updated weights to: {save_ckpt_dir}")

    if debug:
        r2bc_debug_print("[peract_update] update_from_disk_buffer summary:")
        r2bc_debug_print("  num_success:", stats["num_success"])
        r2bc_debug_print("  num_skipped:", stats["num_skipped"])
        r2bc_debug_print("  num_failed:", stats["num_failed"])
        r2bc_debug_print("  losses:", stats["losses"])
        r2bc_debug_print("  arms:", stats["arms"])

        _print_module_stats_block(
            "  module_stats_first:",
            stats.get("module_stats_first", []),
        )
        _print_module_stats_block(
            "  module_stats_last:",
            stats.get("module_stats_last", []),
        )

    return stats