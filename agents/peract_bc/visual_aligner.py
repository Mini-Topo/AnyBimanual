import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scale_stat(name, x):
    if not torch.is_tensor(x):
        print(f"[scale debug] {name}: non-tensor {type(x)}")
        return
    xd = x.detach().float()
    std = xd.std(unbiased=False).item()
    print(
        f"[scale debug] {name}: "
        f"shape={tuple(xd.shape)} "
        f"mean={xd.mean().item():+.4e} "
        f"std={std:+.4e} "
        f"min={xd.min().item():+.4e} "
        f"max={xd.max().item():+.4e}"
    )


class VisualAligner(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, mask_dim=128, gate_gain=0.3):
        super(VisualAligner, self).__init__()

        # Normalize voxel tokens before computing gates.
        # The original PerAct voxel feature scale can be large, so Conv1d directly
        # on raw features easily creates huge multiplicative masks.
        self.input_norm = nn.LayerNorm(input_dim)

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.conv_res1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv_res2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.conv2_right = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=mask_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv2_left = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=mask_dim,
            kernel_size=3,
            padding=1,
        )

        self.activation = nn.ReLU()
        self.gate_gain = gate_gain

    def forward(self, ins):
        """
        Args:
            ins: [B, N, D], e.g. [B, 8000, 128]

        Returns:
            masked_ins_right, masked_ins_left: [B, N, D]

        Instead of an unconstrained ReLU mask, use a bounded residual gate:
            gate = 1 + gate_gain * tanh(raw_gate)

        With gate_gain=0.1, gate is in [0.9, 1.1].
        This preserves the pretrained PerAct feature scale while still allowing
        the aligner to modulate voxel tokens.
        """
        debug_scale = os.environ.get("R2BC_SCALE_DEBUG", "0") == "1"

        if debug_scale:
            _scale_stat("visual_aligner/input_ins_BND", ins)

        ins_orig = ins
        ins_norm = self.input_norm(ins_orig)

        if debug_scale:
            _scale_stat("visual_aligner/input_norm_BND", ins_norm)

        x = ins_norm.transpose(1, 2)  # [B, D, N]

        features = self.activation(self.conv1(x))

        residual = features
        features = self.activation(self.conv_res1(features))
        features = self.conv_res2(features)
        features = features + residual

        raw_right = self.conv2_right(features)
        raw_left = self.conv2_left(features)

        gate_right = 1.0 + self.gate_gain * torch.tanh(raw_right)
        gate_left = 1.0 + self.gate_gain * torch.tanh(raw_left)

        gate_right = gate_right.transpose(1, 2)
        gate_left = gate_left.transpose(1, 2)

        masked_ins_right = ins_orig * gate_right
        masked_ins_left = ins_orig * gate_left

        if debug_scale:
            _scale_stat("visual_aligner/features_BDN", features)
            _scale_stat("visual_aligner/raw_right_BDN", raw_right)
            _scale_stat("visual_aligner/raw_left_BDN", raw_left)
            _scale_stat("visual_aligner/gate_right_BND", gate_right)
            _scale_stat("visual_aligner/gate_left_BND", gate_left)
            _scale_stat("visual_aligner/masked_right_BND", masked_ins_right)
            _scale_stat("visual_aligner/masked_left_BND", masked_ins_left)

        return masked_ins_right, masked_ins_left
