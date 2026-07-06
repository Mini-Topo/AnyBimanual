# Standard Library
import os
import sys
from os.path import join, dirname, abspath

# pygame / SDL noise reduction
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

# Project paths
PROJECT_ROOT = dirname(abspath(__file__))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))

# Third Party
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning

# Local
from agents.agent_factory import create_agent
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.observation_utils import create_obs_config


def _get_type(x):
    arr = np.asarray(x)
    if arr.dtype == np.float64:
        return np.float32
    return arr.dtype


def make_peract_obs_config(cfg):
    camera_names = list(cfg.rlbench.cameras)
    camera_resolution = list(cfg.rlbench.camera_resolution)

    print("[debug] camera_names:", camera_names)
    print("[debug] camera_resolution:", camera_resolution)

    return create_obs_config(
        camera_names=camera_names,
        camera_resolution=camera_resolution,
        method_name=cfg.method.name,
        robot_name="bimanual",
    )


def make_action_mode():
    return BimanualMoveArmThenGripper(
        BimanualEndEffectorPoseViaPlanning(),
        BimanualDiscrete(),
    )


def make_prepped_data(obs, timesteps, device):
    """
    third_party/YARR/yarr/utils/rollout_generator.py と同じ形にする。

    obs[key] -> [timesteps, ...]
             -> [batch=1, timesteps, ...]
             -> torch.Tensor
    """
    obs_history = {
        k: [np.array(v, dtype=_get_type(v))] * timesteps
        for k, v in obs.items()
    }

    prepped_data = {
        k: torch.tensor(np.array(v)[None], device=device)
        for k, v in obs_history.items()
    }

    return prepped_data


def print_obs_shapes(title, obs):
    print(f"\n[debug] {title}")
    for k, v in obs.items():
        arr = np.asarray(v)
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")


def print_tensor_shapes(title, data):
    print(f"\n[debug] {title}")
    for k, v in data.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # -------------------------
    # agent config
    # -------------------------
    cfg.method.name = "PERACT_BC"
    cfg.method.agent_type = "independent"
    cfg.method.robot_name = "bimanual"

    cfg.framework.anybimanual = True
    cfg.framework.checkpoint_name_prefix = "checkpoint"

    cfg.ddp.num_devices = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[debug] method:", cfg.method.name)
    print("[debug] agent_type:", cfg.method.agent_type)
    print("[debug] anybimanual:", cfg.framework.anybimanual)
    print("[debug] checkpoint_name_prefix:", cfg.framework.checkpoint_name_prefix)
    print("[debug] device:", device)

    # -------------------------
    # load agent
    # -------------------------
    print("[debug] creating agent...")
    agent = create_agent(cfg)

    print("[debug] building agent...")
    agent.build(training=True, device=device)

    ckpt_dir = (
        "/home/tappei-m/Project/AnyBimanual_checkpoints/"
        "PERACT_BC_pure_peract_600k"
    )

    print("[debug] loading weights from:", ckpt_dir)
    agent.load_weights(ckpt_dir)
    print("[debug] load success!")

    def group_name(param_name: str) -> str:
        if "skill_manager" in param_name:
            return "skill_manager"
        if "visual_aligner" in param_name:
            return "visual_aligner"
        if "trans_decoder" in param_name:
            return "trans_decoder"
        if "rot_grip_collision_ff" in param_name:
            return "rot_grip_collision_ff"
        if "rot_grip_dense" in param_name:
            return "rot_grip_dense"
        if "final_conv3d" in param_name:
            return "final_conv3d"
        return "peract_core_or_other"

    def print_param_groups(module, prefix="module"):

        if not hasattr(module, "named_parameters"):
            print(f"[debug] {prefix} has no named_parameters()")
            return

        stats = {}

        for name, p in module.named_parameters():
            g = group_name(name)

            if g not in stats:
                stats[g] = {
                    "num_tensors": 0,
                    "num_params": 0,
                    "trainable_tensors": 0,
                    "trainable_params": 0,
                    "trainable_examples": [],
                    "frozen_examples": [],
                }

            stats[g]["num_tensors"] += 1
            stats[g]["num_params"] += p.numel()

            if p.requires_grad:
                stats[g]["trainable_tensors"] += 1
                stats[g]["trainable_params"] += p.numel()
                if len(stats[g]["trainable_examples"]) < 8:
                    stats[g]["trainable_examples"].append((name, tuple(p.shape)))
            else:
                if len(stats[g]["frozen_examples"]) < 3:
                    stats[g]["frozen_examples"].append((name, tuple(p.shape)))

        print(f"\n[debug] ===== PARAM GROUP SUMMARY: {prefix} =====")
        for g, s in stats.items():
            print(f"\n[group] {g}")
            print(f"  tensors:          {s['num_tensors']}")
            print(f"  params:           {s['num_params']}")
            print(f"  trainable tensors:{s['trainable_tensors']}")
            print(f"  trainable params: {s['trainable_params']}")

            for name, shape in s["trainable_examples"]:
                print(f"    [trainable] {name} {shape}")

            if s["trainable_tensors"] == 0:
                for name, shape in s["frozen_examples"]:
                    print(f"    [frozen example] {name} {shape}")

    def print_module_tree_keywords(module, prefix="module"):
        if not hasattr(module, "named_modules"):
            print(f"[debug] {prefix} has no named_modules()")
            return

        print(f"\n[debug] ===== MODULE TREE KEYWORDS: {prefix} =====")
        keywords = [
            "skill_manager",
            "visual_aligner",
            "trans_decoder",
            "rot_grip",
            "final_conv3d",
            "perceiver",
            "encoder",
            "decoder",
        ]

        for name, m in module.named_modules():
            if any(k in name for k in keywords):
                print(f"  {name}: {type(m)}")

    def inspect_single_agent(single_agent, prefix):
        print(f"\n[debug] ===== INSPECT {prefix} =====")
        print(f"[debug] {prefix} type:", type(single_agent))
        print(f"[debug] {prefix} dict keys:", single_agent.__dict__.keys())

        print_param_groups(single_agent, prefix)

        # PreprocessAgent の中身を掘る
        if hasattr(single_agent, "_pose_agent"):
            pose_agent = single_agent._pose_agent
            print(f"\n[debug] {prefix}._pose_agent type:", type(pose_agent))
            print(f"[debug] {prefix}._pose_agent dict keys:", pose_agent.__dict__.keys())

            print_param_groups(pose_agent, f"{prefix}._pose_agent")

            if hasattr(pose_agent, "_qattention_agents"):
                print(f"\n[debug] {prefix}._pose_agent._qattention_agents type:", type(pose_agent._qattention_agents))
                print(f"[debug] {prefix}._pose_agent._qattention_agents len:", len(pose_agent._qattention_agents))

                for i, qa in enumerate(pose_agent._qattention_agents):
                    qa_prefix = f"{prefix}._pose_agent._qattention_agents[{i}]"

                    print(f"\n[debug] ===== INSPECT {qa_prefix} =====")
                    print(f"[debug] {qa_prefix} type:", type(qa))
                    print(f"[debug] {qa_prefix} dict keys:", qa.__dict__.keys())

                    print_param_groups(qa, qa_prefix)

                    if hasattr(qa, "_q"):
                        print(f"[debug] {qa_prefix}._q type:", type(qa._q))
                        print_param_groups(qa._q, f"{qa_prefix}._q")

                        if hasattr(qa._q, "_qnet"):
                            print(f"[debug] {qa_prefix}._q._qnet type:", type(qa._q._qnet))
                            print_param_groups(qa._q._qnet, f"{qa_prefix}._q._qnet")
                            print_module_tree_keywords(
                                qa._q._qnet,
                                f"{qa_prefix}._q._qnet",
                            )

            if hasattr(pose_agent, "_q"):
                print(f"[debug] {prefix}._pose_agent._q type:", type(pose_agent._q))
                print_param_groups(pose_agent._q, f"{prefix}._pose_agent._q")

                if hasattr(pose_agent._q, "_qnet"):
                    print(f"[debug] {prefix}._pose_agent._q._qnet type:", type(pose_agent._q._qnet))
                    print_param_groups(pose_agent._q._qnet, f"{prefix}._pose_agent._q._qnet")
                    print_module_tree_keywords(
                        pose_agent._q._qnet,
                        f"{prefix}._pose_agent._q._qnet",
                    )

        # 念のため、single_agent 直下に _q がある場合も見る
        if hasattr(single_agent, "_q"):
            print(f"[debug] {prefix}._q type:", type(single_agent._q))
            print_param_groups(single_agent._q, f"{prefix}._q")

            if hasattr(single_agent._q, "_qnet"):
                print(f"[debug] {prefix}._q._qnet type:", type(single_agent._q._qnet))
                print_param_groups(single_agent._q._qnet, f"{prefix}._q._qnet")
                print_module_tree_keywords(single_agent._q._qnet, f"{prefix}._q._qnet")
    
    print("\n[debug] ===== TOP LEVEL AGENT =====")
    print("[debug] agent type:", type(agent))
    print("[debug] agent dict keys:", agent.__dict__.keys())

    if hasattr(agent, "right_agent"):
        inspect_single_agent(agent.right_agent, "right_agent")

    if hasattr(agent, "left_agent"):
        inspect_single_agent(agent.left_agent, "left_agent")

    return

if __name__ == "__main__":
    main()