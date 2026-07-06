import torch
import hydra
from omegaconf import DictConfig

from agents.agent_factory import create_agent


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 明示的に上書き
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

    print("[debug] creating agent...")
    agent = create_agent(cfg)

    print("[debug] building agent...")
    agent.build(training=False, device=device)

    ckpt_dir = (
        "/home/tappei-m/Project/AnyBimanual_checkpoints/"
        "PERACT_BC_leader_as_independent"
    )

    print("[debug] loading weights from:", ckpt_dir)
    agent.load_weights(ckpt_dir)

    print("[debug] load success!")


if __name__ == "__main__":
    main()