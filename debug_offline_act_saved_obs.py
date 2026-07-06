import sys
import pickle
from os.path import abspath, join

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
from r2bc.peract_update import obs_to_replay_inputs

CKPT_DIR = "/home/tappei-m/Project/AnyBimanual_checkpoints/R2BC_async_2ep_update_150"

PKL_PATH = (
    "data/debug_collect_gripper_fix_rollout/"
    "bimanual_lift_long_block/right_human/episode_000000.pkl"
)
TARGET_ARM = "left"


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


def max_abs_diff(a, b):
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    return float((a - b).abs().max().item())


def main():
    device = torch.device("cuda:0")

    cfg = load_cfg()

    policy = AnyBimanualPerActPolicy(
        cfg=cfg,
        ckpt_dir=CKPT_DIR,
        device="cuda:0",
        timesteps=1,
        deterministic=True,
    )

    with open(PKL_PATH, "rb") as f:
        ep = pickle.load(f)

    STEP_INDEX = 1
    obs = ep["steps"][STEP_INDEX]["obs"]

    print("STEP_INDEX:", STEP_INDEX)
    print("GT target xyz:", ep["steps"][STEP_INDEX]["target_action_9d"][:3])
    print("GT grip:", ep["steps"][STEP_INDEX]["target_action_9d"][7])

    print("=" * 80)
    print("Offline act on saved obs")

    with torch.no_grad():
        full_action = policy.act_full(obs, STEP_INDEX)

    full_action = np.asarray(full_action, dtype=np.float32)

    right_pred = full_action[:9]
    left_pred = full_action[9:18]
    pred = left_pred if TARGET_ARM == "left" else right_pred

    gt = ep["steps"][STEP_INDEX]["target_action_9d"].astype(np.float32)

    info = ep["steps"][STEP_INDEX]["info"]

    logged_right_policy = np.asarray(info["right_policy_action_9d"], dtype=np.float32)
    logged_left_policy = np.asarray(info["left_policy_action_9d"], dtype=np.float32)

    print("=" * 80)
    print("Compare with logged rollout policy actions")
    print("STEP_INDEX:", STEP_INDEX)
    print("TARGET_ARM:", TARGET_ARM)

    print("logged_right_policy:", logged_right_policy)
    print("logged_left_policy :", logged_left_policy)

    print("pred_right_from_act:", right_pred)
    print("pred_left_from_act :", left_pred)

    print("right_policy_xyz_err:",
        float(np.linalg.norm(right_pred[:3] - logged_right_policy[:3])))
    print("left_policy_xyz_err :",
        float(np.linalg.norm(left_pred[:3] - logged_left_policy[:3])))

    print("logged_left_xyz:", logged_left_policy[:3])
    print("pred_left_xyz  :", left_pred[:3])

    print("TARGET_ARM:", TARGET_ARM)
    print("full_action shape:", full_action.shape)
    print("right_pred:", right_pred)
    print("left_pred :", left_pred)
    print("gt        :", gt)

    print("gt_xyz    :", gt[:3])
    print("pred_xyz  :", pred[:3])
    print("xyz_err   :", float(np.linalg.norm(pred[:3] - gt[:3])))

    print("gt_quat   :", gt[3:7])
    print("pred_quat :", pred[3:7])

    print("gt_grip/pred_grip:", float(gt[7]), float(pred[7]))
    print("gt_col/pred_col  :", float(gt[8]), float(pred[8]))

    # ------------------------------------------------------------
    # act path: make_prepped_data -> BimanualAgent と同じ left prefix 削り
    # ------------------------------------------------------------
    prepped = policy.make_prepped_data(obs)

    left_observation = {}
    for k, v in prepped.items():
        if "rgb" in k or "point_cloud" in k or "camera" in k:
            left_observation[k] = v
        elif "left_" in k:
            left_observation[k[5:]] = v
        elif "right_" in k:
            pass
        else:
            left_observation[k] = v

    # qattention act 内部と同じように [0] slice する
    act_inputs = {}

    for k, v in left_observation.items():
        if k.endswith("_rgb"):
            # Match PreprocessAgent._norm_rgb_(zero_mean)
            act_inputs[k] = ((v[0].float().to(device) / 255.0) * 2.0 - 1.0)
        elif k.endswith("_point_cloud") or k == "low_dim_state":
            act_inputs[k] = v[0].float().to(device)

    # ------------------------------------------------------------
    # update path
    # ------------------------------------------------------------
    # left_agent は QAttentionStackAgent のはず
    qagent = policy.agent.left_agent._pose_agent._qattention_agents[0]

    update_inputs = obs_to_replay_inputs(
        obs=obs,
        arm=TARGET_ARM,
        clip_qagent=qagent,
        device=device,
        debug=False,
    )

    print("=" * 80)
    print("Compare update path vs act path inputs")
    print("PKL_PATH:", PKL_PATH)
    print("CKPT_DIR:", CKPT_DIR)
    print("TARGET_ARM:", TARGET_ARM)
    print("=" * 80)

    compare_keys = []
    for k in update_inputs.keys():
        if k.endswith("_rgb") or k.endswith("_point_cloud") or k == "low_dim_state":
            compare_keys.append(k)

    for k in sorted(compare_keys):
        print("-" * 80)
        print(k)

        if k not in act_inputs:
            print("[missing in act_inputs]")
            continue

        u = update_inputs[k]
        a = act_inputs[k]

        print("update shape:", tuple(u.shape), "dtype:", u.dtype, "device:", u.device)
        print("act    shape:", tuple(a.shape), "dtype:", a.dtype, "device:", a.device)

        if tuple(u.shape) != tuple(a.shape):
            print("[shape mismatch]")

        try:
            print("max_abs_diff:", max_abs_diff(u, a))
            print("update min/max:", float(u.min().item()), float(u.max().item()))
            print("act    min/max:", float(a.min().item()), float(a.max().item()))
        except Exception as e:
            print("compare failed:", repr(e))

    print("-" * 80)
    print("lang_goal_tokens")
    print("prepped shape:", tuple(prepped["lang_goal_tokens"].shape))
    print("left obs shape:", tuple(left_observation["lang_goal_tokens"].shape))
    print(
        "tokens first 10:",
        left_observation["lang_goal_tokens"].detach().cpu().flatten()[:10].numpy(),
    )

    print("-" * 80)
    print("lang embeddings from update path")
    for k in ["lang_goal_emb", "lang_token_embs"]:
        v = update_inputs[k]
        print(k, tuple(v.shape), v.dtype, v.device, "min/max:", float(v.min()), float(v.max()))

    print("-" * 80)
    print("compare language embeddings: update vs act-style")

    with torch.no_grad():
        act_tokens = left_observation["lang_goal_tokens"].long().to(device)
        act_lang_goal_emb, act_lang_token_embs = qagent._clip_rn50.encode_text_with_embeddings(
            act_tokens[0]
        )

    act_lang_goal_emb = act_lang_goal_emb.to(device)
    act_lang_token_embs = act_lang_token_embs.to(device)

    print("act_lang_goal_emb", tuple(act_lang_goal_emb.shape), act_lang_goal_emb.dtype)
    print("act_lang_token_embs", tuple(act_lang_token_embs.shape), act_lang_token_embs.dtype)

    print("=" * 80)
    print("Check same obs with different step_id")

    for sid in [0, 1, 2]:
        with torch.no_grad():
            fa = policy.act_full(obs, sid)

        fa = np.asarray(fa, dtype=np.float32)
        rp = fa[:9]
        lp = fa[9:18]

        print("-" * 80)
        print("sid:", sid)
        print("right_pred:", rp)
        print("left_pred :", lp)

    print(
        "lang_goal_emb max_abs_diff:",
        max_abs_diff(update_inputs["lang_goal_emb"], act_lang_goal_emb),
    )
    print(
        "lang_token_embs max_abs_diff:",
        max_abs_diff(update_inputs["lang_token_embs"], act_lang_token_embs),
    )
    print(
        "update lang_goal min/max:",
        float(update_inputs["lang_goal_emb"].min()),
        float(update_inputs["lang_goal_emb"].max()),
    )
    print(
        "act lang_goal min/max:",
        float(act_lang_goal_emb.min()),
        float(act_lang_goal_emb.max()),
    )


if __name__ == "__main__":
    main()