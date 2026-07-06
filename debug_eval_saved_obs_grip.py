import os
import sys
import pickle
from os.path import join, dirname, abspath

import numpy as np
import torch

try:
    from hydra import compose, initialize_config_dir
except ImportError:
    from hydra.experimental import compose, initialize_config_dir

PROJECT_ROOT = abspath(".")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "RLBench"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "PyRep"))
sys.path.insert(0, join(PROJECT_ROOT, "third_party", "YARR"))

from r2bc.anybimanual_peract_policy import AnyBimanualPerActPolicy


def load_cfg():
    config_dir = join(PROJECT_ROOT, "conf")
    overrides = [
        "method=PERACT_BC",
        "method.agent_type=independent",
        "framework.anybimanual=True",
        "framework.checkpoint_name_prefix=checkpoint",
        "ddp.num_devices=1",
    ]

    try:
        hydra_ctx = initialize_config_dir(
            config_dir=config_dir,
            version_base=None,
        )
    except TypeError:
        hydra_ctx = initialize_config_dir(
            config_dir=config_dir,
        )

    with hydra_ctx:
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

    return cfg


def main():
    ckpt_dir = "/tmp/r2bc_debug_update_act_style_coords"
    pkl_path = (
        "data/r2bc_keyframe_left_12ep/"
        "bimanual_lift_long_block/left_human/episode_000000.pkl"
    )

    cfg = load_cfg()

    policy = AnyBimanualPerActPolicy(
        cfg=cfg,
        ckpt_dir=ckpt_dir,
        device="cuda:0",
        timesteps=1,
        deterministic=True,
    )

    print("[debug mode] right q training:",
        policy.agent.right_agent._pose_agent._qattention_agents[0]._q.training)
    print("[debug mode] right qnet training:",
        policy.agent.right_agent._pose_agent._qattention_agents[0]._q._qnet.training)

    print("[debug mode] left q training:",
        policy.agent.left_agent._pose_agent._qattention_agents[0]._q.training)
    print("[debug mode] left qnet training:",
        policy.agent.left_agent._pose_agent._qattention_agents[0]._q._qnet.training)

    # DEBUG: force left q to train mode
    left_qagent = policy.agent.left_agent._pose_agent._qattention_agents[0]
    left_qagent._q.train()
    left_qagent._q._qnet.train()

    print("[debug mode after force] left q training:",
        left_qagent._q.training)
    print("[debug mode after force] left qnet training:",
        left_qagent._q._qnet.training)

    # policy.agent.right_agent._pose_agent._qattention_agents[0]._q.train()
    # policy.agent.left_agent._pose_agent._qattention_agents[0]._q.train()

    # print("[debug mode after force] right q training:",
    #     policy.agent.right_agent._pose_agent._qattention_agents[0]._q.training)
    # print("[debug mode after force] right qnet training:",
    #     policy.agent.right_agent._pose_agent._qattention_agents[0]._q._qnet.training)

    # print("[debug mode after force] left q training:",
    #     policy.agent.left_agent._pose_agent._qattention_agents[0]._q.training)
    # print("[debug mode after force] left qnet training:",
    #     policy.agent.left_agent._pose_agent._qattention_agents[0]._q._qnet.training)

    with open(pkl_path, "rb") as f:
        ep = pickle.load(f)

    print("[offline eval] pkl:", pkl_path)
    print("[offline eval] num steps:", len(ep["steps"]))

    for i, step in enumerate(ep["steps"]):
        obs = step["obs"]
        gt = np.asarray(step["target_action_9d"], dtype=np.float32)

        print("=" * 80)
        print("[offline eval] step:", i)
        print("[offline eval] target_arm:", step.get("target_arm"))
        print("[offline eval] GT left target xyz:", gt[:3])
        print("[offline eval] GT grip_open:", gt[7])

        # ------------------------------------------------------------
        # left-only act debug: bypass BimanualAgent right->left call
        # ------------------------------------------------------------
        prepped = policy.make_prepped_data(obs)

        left_observation = {}
        for k, v in prepped.items():
            if "rgb" in k or "point_cloud" in k or "camera" in k:
                # camera inputs are shared, no right_/left_ prefix
                left_observation[k] = v
            elif k.startswith("left_"):
                # left_low_dim_state -> low_dim_state
                left_observation[k[5:]] = v
            elif k.startswith("right_"):
                # discard right proprio keys
                pass
            else:
                # lang_goal_tokens etc.
                left_observation[k] = v

        left_only_res = policy.agent.left_agent.act(
            step=i,
            observation=left_observation,
            deterministic=True,
            arm="left",
        )

        left_only_action = np.asarray(left_only_res.action, dtype=np.float32)

        print("[left-only act] pred left grip:", left_only_action[7])
        print("[left-only act] pred left xyz:", left_only_action[:3])

        # qattention_stack_agent.py に qstack debug print が入っていれば、
        # ここで right/left の rgai と grip_idx が表示される
        action18 = policy.act_full(obs, i)
        action18 = np.asarray(action18, dtype=np.float32)

        right = action18[:9]
        left = action18[9:18]

        print("[offline eval] pred right grip:", right[7])
        print("[offline eval] pred left  grip:", left[7])
        print("[offline eval] pred left  xyz:", left[:3])


if __name__ == "__main__":
    main()