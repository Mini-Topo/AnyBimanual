import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch3d import transforms as torch3d_tf
from yarr.agents.agent import (
    Agent,
    ActResult,
    ScalarSummary,
    HistogramSummary,
    ImageSummary,
    Summary,
)
import matplotlib.pyplot as plt
import PIL.Image as Image
import wandb
import io
from termcolor import colored, cprint
from helpers import utils
from helpers.utils import visualise_voxel, stack_on_channel
from voxel.voxel_grid import VoxelGrid
from einops import rearrange
from helpers.clip.core.clip import build_model, load_clip

import transformers
from helpers.optim.lamb import Lamb

from torch.nn.parallel import DistributedDataParallel as DDP

def r2bc_debug_enabled() -> bool:
    return os.environ.get("R2BC_VERBOSE_DEBUG", "0") == "1"


def r2bc_debug_print(*args, **kwargs):
    if r2bc_debug_enabled():
        print(*args, **kwargs)


class QFunction(nn.Module):
    def __init__(
        self,
        perceiver_encoder: nn.Module,
        voxelizer: VoxelGrid,
        bounds_offset: float,
        rotation_resolution: float,
        device,
        training,
        use_ddp: bool = False,
    ):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)

        # distributed training
        if training and use_ddp:
            device_id = device.index if isinstance(device, torch.device) else device
            self._qnet = DDP(
                self._qnet,
                device_ids=[device_id],
                find_unused_parameters=True,
            )

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(
                torch.split(
                    q_rot_grip[:, :-2], int(360 // self._rotation_resolution), dim=1
                ),
                dim=1,
            )
            rot_and_grip_indicies = torch.cat(
                [
                    q_rot[:, 0:1].argmax(-1),
                    q_rot[:, 1:2].argmax(-1),
                    q_rot[:, 2:3].argmax(-1),
                    q_rot_grip[:, -2:].argmax(-1, keepdim=True),
                ],
                -1,
            )
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(
        self,
        rgb_pcd,
        proprio,
        pcd,
        lang_goal_emb,
        lang_token_embs,
        bounds=None,
        prev_bounds=None,
        prev_layer_voxel_grid=None,
        arm=None,
    ):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1
        )

        # batch bounds if necessary
        if bounds is not None and bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        if prev_bounds is not None and prev_bounds.shape[0] != b:
            prev_bounds = prev_bounds.repeat(b, 1)

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds
        )

        # swap to channels first
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # forward pass
        q_trans, q_rot_and_grip, q_ignore_collisions, L_skill, L_voxel = self._qnet(
            voxel_grid,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_bounds,
            arm=arm,
        )

        return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid, L_skill, L_voxel


class QAttentionPerActBCAgent(Agent):
    def __init__(
        self,
        layer: int,
        coordinate_bounds: list,
        perceiver_encoder: nn.Module,
        camera_names: list,
        batch_size: int,
        voxel_size: int,
        bounds_offset: float,
        voxel_feature_size: int,
        image_crop_size: int,
        num_rotation_classes: int,
        rotation_resolution: float,
        lr: float = 0.0001,
        lr_scheduler: bool = False,
        training_iterations: int = 100000,
        num_warmup_steps: int = 20000,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        grip_loss_weight: float = 1.0,
        collision_loss_weight: float = 1.0,
        include_low_dim_state: bool = False,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
        transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "adam",
        num_devices: int = 1,
        checkpoint_name_prefix=None,
        anybimanual = False,
        cfg=None,
    ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        checkpoint_name_prefix = checkpoint_name_prefix or "QAttentionAgent"
        self._name = f"{checkpoint_name_prefix}_layer_{self._layer}"
        self.anybimanual = anybimanual
        self.cfg=cfg


    def build(self, training: bool, device: torch.device = None):
        self._training = training

        if device is None:
            device = torch.device("cpu")

        self._device = device

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        use_ddp = self._num_devices > 1

        self._q = (
            QFunction(
                self._perceiver_encoder,
                self._voxelizer,
                self._bounds_offset,
                self._rotation_resolution,
                device,
                training,
                use_ddp=use_ddp,
            )
            .to(device)
            .train(training)
        )

        grid_for_crop = (
            torch.arange(0, self._image_crop_size, device=device)
            .unsqueeze(0)
            .repeat(self._image_crop_size, 1)
            .unsqueeze(-1)
        )
        self._grid_for_crop = torch.cat(
            [grid_for_crop.transpose(1, 0), grid_for_crop], dim=2
        ).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(
            self._coordinate_bounds, device=device
        ).unsqueeze(0)

        if self._training:

            if self.anybimanual:
                for name, p in self._q.named_parameters():
                    p.requires_grad = False

                unfreeze_trans_head = os.environ.get("R2BC_UNFREEZE_TRANS_HEAD", "0") == "1"
                unfreeze_rot_grip_head = os.environ.get("R2BC_UNFREEZE_ROT_GRIP_HEAD", "0") == "1"
                unfreeze_rot_grip_dense = os.environ.get("R2BC_UNFREEZE_ROT_GRIP_DENSE", "0") == "1"
                unfreeze_final_conv = os.environ.get("R2BC_UNFREEZE_FINAL_CONV", "0") == "1"

                for name, p in self._q.named_parameters():
                    lname = name.lower()

                    is_anybimanual_module = (
                        "skill_manager" in lname or "visual_aligner" in lname
                    )

                    is_trans_head = (
                        unfreeze_trans_head
                        and "trans_decoder" in lname
                    )

                    is_rot_grip_head = (
                        unfreeze_rot_grip_head
                        and "rot_grip_collision_ff" in lname
                    )

                    is_rot_grip_dense = (
                        unfreeze_rot_grip_dense
                        and ("dense0" in lname or "dense1" in lname)
                    )

                    is_final_conv = (
                        unfreeze_final_conv
                        and "final.conv3d" in lname
                    )

                    if (
                        is_anybimanual_module
                        or is_trans_head
                        or is_rot_grip_head
                        or is_rot_grip_dense
                        or is_final_conv
                    ):
                        p.requires_grad = True
                        logging.info("[R2BC trainable] %s %s", name, tuple(p.shape))

                trainable_params = [p for p in self._q.parameters() if p.requires_grad]

                logging.info(
                    "[R2BC trainable params] %d",
                    sum(p.numel() for p in trainable_params)
                )
            else:
                trainable_params = self._q.parameters()

            # optimizer
            if self._optimizer_type == "lamb":
                self._optimizer = Lamb(
                    trainable_params,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )

            elif self._optimizer_type == "adam":
                self._optimizer = torch.optim.Adam(
                    trainable_params,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )

            else:
                raise Exception("Unknown optimizer type")

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = (
                    transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                        self._optimizer,
                        num_warmup_steps=self._num_warmup_steps,
                        num_training_steps=self._training_iterations,
                        num_cycles=self._training_iterations // 10000,
                    )
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros(
                (
                    self._batch_size,
                    1,
                    self._voxel_size,
                    self._voxel_size,
                    self._voxel_size,
                ),
                dtype=int,
                device=device,
            )
            self._action_rot_x_one_hot_zeros = torch.zeros(
                (self._batch_size, self._num_rotation_classes), dtype=int, device=device
            )
            self._action_rot_y_one_hot_zeros = torch.zeros(
                (self._batch_size, self._num_rotation_classes), dtype=int, device=device
            )
            self._action_rot_z_one_hot_zeros = torch.zeros(
                (self._batch_size, self._num_rotation_classes), dtype=int, device=device
            )
            self._action_grip_one_hot_zeros = torch.zeros(
                (self._batch_size, 2), dtype=int, device=device
            )
            self._action_ignore_collisions_one_hot_zeros = torch.zeros(
                (self._batch_size, 2), dtype=int, device=device
            )

            # print total params
            logging.info(
                "# Q Params: %d"
                % sum(
                    p.numel()
                    for name, p in self._q.named_parameters()
                    if p.requires_grad and "clip" not in name
                )
            )
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0, h - self._image_crop_size
        )
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode="nearest", align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample["%s_rgb" % n]
            pcd = replay_sample["%s_point_cloud" % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation["%s_rgb" % n]
            pcd = observation["%s_point_cloud" % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (
            voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2]
        )[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[
            ..., 0
        ]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(
            torch.split(
                rot_grip_q[:, :-2], int(360 // self._rotation_resolution), dim=1
            ),
            dim=1,
        )  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [
                q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
                q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
                q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
                q_grip.gather(1, rot_and_grip_idx[:, 3:4]),
            ],
            -1,
        )
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[
            :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
        ]
        q_rot_y_flat = q_rot_grip[
            :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
        ]
        q_rot_z_flat = q_rot_grip[
            :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
        ]
        q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes :]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat(
            [
                q_rot_x_flat_softmax,
                q_rot_y_flat_softmax,
                q_rot_z_flat_softmax,
                q_grip_flat_softmax,
            ],
            dim=1,
        )

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def update(
        self,
        step: int,
        replay_sample: dict,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        loss_scale: float = 1.0,
    ) -> dict:
        action_trans = replay_sample["trans_action_indicies"][
            :, self._layer * 3 : self._layer * 3 + 3
        ].int()
        action_rot_grip = replay_sample["rot_grip_action_indicies"].int()
        action_gripper_pose = replay_sample["gripper_pose"]
        action_ignore_collisions = replay_sample["ignore_collisions"].int()
        lang_goal_emb = replay_sample["lang_goal_emb"].float()
        lang_token_embs = replay_sample["lang_token_embs"].float()
        prev_layer_voxel_grid = replay_sample.get("prev_layer_voxel_grid", None)
        prev_layer_bounds = replay_sample.get("prev_layer_bounds", None)
        device = self._device
        rank = device
        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample["attention_coordinate_layer_%d" % (self._layer - 1)]
            bounds = torch.cat(
                [cp - self._bounds_offset, cp + self._bounds_offset], dim=1
            )

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample["low_dim_state"]

        obs, pcd = self._preprocess_inputs(replay_sample)

        if "target_arm_id" in replay_sample:
            target_arm_id = int(replay_sample["target_arm_id"][0].detach().cpu().item())
            arm = "right" if target_arm_id == 0 else "left"
        elif "target_arm" in replay_sample:
            arm = replay_sample["target_arm"]
        else:
            # fallback: original behavior
            if proprio.shape[-1] == 4:
                arm = "right"
            else:
                arm = "left"

        if step == 0:
            r2bc_debug_print("[debug update] arm:", arm)

        # batch size
        bs = pcd[0].shape[0]

        # SE(3) augmentation of point clouds and actions
        if self._transform_augmentation:
            from voxel import augmentation
            action_trans, action_rot_grip, pcd = augmentation.apply_se3_augmentation(
                pcd,
                action_gripper_pose,
                action_trans,
                action_rot_grip,
                bounds,
                self._layer,
                self._transform_augmentation_xyz,
                self._transform_augmentation_rpy,
                self._transform_augmentation_rot_resolution,
                self._voxel_size,
                self._rotation_resolution,
                self._device,
            )

        # forward pass
        q_trans, q_rot_grip, q_collision, voxel_grid, L_skill, L_voxel = self._q(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
            arm=arm,
        )

        # argmax to choose best action
        (
            coords,
            rot_and_grip_indicies,
            ignore_collision_indicies,
        ) = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)

        # ------------------------------------------------------------
        # DEBUG: update path の rot/grip prediction を直接見る
        # ------------------------------------------------------------
        debug_gt_rot_grip = None
        debug_pred_rot_grip_raw = None
        debug_pred_rot_grip_act_style = None
        debug_gt_grip = None
        debug_pred_grip_raw = None
        debug_pred_grip_act_style = None
        debug_pred_voxel_raw = None
        debug_pred_voxel_act_style = None

        if q_rot_grip is not None:
            with torch.no_grad():
                # update path で使っている raw q からの argmax
                debug_gt_rot_grip = action_rot_grip[0].detach().cpu()
                debug_pred_rot_grip_raw = rot_and_grip_indicies[0].detach().cpu()
                debug_pred_voxel_raw = coords[0].detach().cpu()

                debug_gt_grip = int(debug_gt_rot_grip[-1].item())
                debug_pred_grip_raw = int(debug_pred_rot_grip_raw[-1].item())

                # act() path と同じように softmax 後に argmax したもの
                q_trans_act_style = self._softmax_q_trans(q_trans.detach())
                q_rot_grip_act_style = self._softmax_q_rot_grip(q_rot_grip.detach())
                q_collision_act_style = (
                    self._softmax_ignore_collision(q_collision.detach())
                    if q_collision is not None
                    else q_collision
                )

                (
                    debug_coords_act_style,
                    debug_rot_grip_act_style,
                    debug_collision_act_style,
                ) = self._q.choose_highest_action(
                    q_trans_act_style,
                    q_rot_grip_act_style,
                    q_collision_act_style,
                )

                debug_pred_voxel_act_style = (
                    debug_coords_act_style[0].detach().cpu()
                )

                debug_pred_rot_grip_act_style = (
                    debug_rot_grip_act_style[0].detach().cpu()
                )
                debug_pred_grip_act_style = int(
                    debug_pred_rot_grip_act_style[-1].item()
                )

                if os.environ.get("R2BC_DEBUG_ROT_GRIP_UPDATE", "0") == "1":
                    debug_every = int(os.environ.get("R2BC_DEBUG_ROT_GRIP_EVERY", "20"))

                    if step < 3 or step % debug_every == 0:
                        r2bc_debug_print(
                            "[update rot/grip debug]",
                            f"step={step}",
                            f"arm={arm}",
                            f"gt_voxel={action_trans[0].detach().cpu().tolist()}",
                            f"pred_voxel_raw={debug_pred_voxel_raw.tolist()}",
                            f"pred_voxel_act_style={debug_pred_voxel_act_style.tolist()}",
                            f"gt_rot_grip={debug_gt_rot_grip.tolist()}",
                            f"pred_raw={debug_pred_rot_grip_raw.tolist()}",
                            f"pred_act_style={debug_pred_rot_grip_act_style.tolist()}",
                            f"gt_grip={debug_gt_grip}",
                            f"pred_grip_raw={debug_pred_grip_raw}",
                            f"pred_grip_act_style={debug_pred_grip_act_style}",
                        )
                
                ####################################################

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0.0, 0.0, 0.0, 0.0

        # translation one-hot
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # translation loss
        q_trans_flat = q_trans.view(bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        if os.environ.get("R2BC_LOG_TRANS_LOGITS", "0") == "1":
            with torch.no_grad():
                for b in range(bs):
                    gt_coord = action_trans[b, :].detach().long()
                    pred_coord_raw = coords[b]

                    if torch.is_tensor(pred_coord_raw):
                        pred_coord = pred_coord_raw.detach().long().to(q_trans.device)
                    else:
                        pred_coord = torch.as_tensor(
                            pred_coord_raw,
                            dtype=torch.long,
                            device=q_trans.device,
                        )

                    # q_trans is usually [B, C, X, Y, Z] with C=1.
                    # Keep this robust in case channel dimension is absent.
                    if q_trans.dim() == 5:
                        gt_logit = q_trans[
                            b, :, gt_coord[0], gt_coord[1], gt_coord[2]
                        ].max()
                        pred_logit = q_trans[
                            b, :, pred_coord[0], pred_coord[1], pred_coord[2]
                        ].max()
                    elif q_trans.dim() == 4:
                        gt_logit = q_trans[
                            b, gt_coord[0], gt_coord[1], gt_coord[2]
                        ]
                        pred_logit = q_trans[
                            b, pred_coord[0], pred_coord[1], pred_coord[2]
                        ]
                    else:
                        gt_logit = torch.tensor(float("nan"), device=q_trans.device)
                        pred_logit = torch.tensor(float("nan"), device=q_trans.device)

                    gt_rank = int((q_trans_flat[b] > gt_logit).sum().item()) + 1
                    pred_rank = int((q_trans_flat[b] > pred_logit).sum().item()) + 1
                    margin_pred_minus_gt = (pred_logit - gt_logit).item()

                    r2bc_debug_print(
                        "[trans logit debug]"
                        f" step={step}"
                        f" b={b}"
                        f" gt={gt_coord.detach().cpu().tolist()}"
                        f" pred={pred_coord.detach().cpu().tolist()}"
                        f" gt_logit={gt_logit.item():.6f}"
                        f" pred_logit={pred_logit.item():.6f}"
                        f" margin_pred_minus_gt={margin_pred_minus_gt:.6f}"
                        f" gt_rank={gt_rank}"
                        f" pred_rank={pred_rank}"
                    )

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = (
                self._action_ignore_collisions_one_hot_zeros.clone()
            )

            for b in range(bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[
                :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
            ]
            q_rot_y_flat = q_rot_grip[
                :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
            ]
            q_rot_z_flat = q_rot_grip[
                :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
            ]
            q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes :]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(
                q_ignore_collisions_flat, action_ignore_collisions_one_hot
            )

        combined_losses = (
            (q_trans_loss * self._trans_loss_weight)
            + (q_rot_loss * self._rot_loss_weight)
            + (q_grip_loss * self._grip_loss_weight)
            + (q_collision_loss * self._collision_loss_weight)
        )

        if os.environ.get("R2BC_TRANS_ONLY", "0") == "1":
            combined_losses = q_trans_loss * self._trans_loss_weight

        total_loss = combined_losses.mean()
        
        if step % 10 == 0 and rank == 0:
            if wandb.run is not None:
                wandb.log({
                    'train/grip_loss': q_grip_loss.mean(),
                    'train/trans_loss': q_trans_loss.mean(),
                    'train/rot_loss': q_rot_loss.mean(),
                    'train/collision_loss': q_collision_loss.mean(),
                    'train/total_loss': total_loss,
                }, step=step)

        if zero_grad:
            self._optimizer.zero_grad()

        (total_loss * loss_scale).backward()

        if step_optimizer:
            self._optimizer.step()

        self._summaries = {
            "losses/total_loss": total_loss,
            "losses/trans_loss": q_trans_loss.mean(),
            "losses/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,
            "losses/collision_loss": q_collision_loss.mean()
            if with_rot_and_grip
            else 0.0,

            "debug/gt_rot_grip": debug_gt_rot_grip,
            "debug/pred_rot_grip_raw": debug_pred_rot_grip_raw,
            "debug/pred_rot_grip_act_style": debug_pred_rot_grip_act_style,
            "debug/gt_grip": debug_gt_grip,
            "debug/pred_grip_raw": debug_pred_grip_raw,
            "debug/pred_grip_act_style": debug_pred_grip_act_style,

            "debug/pred_voxel_raw": debug_pred_voxel_raw,
            "debug/pred_voxel_act_style": debug_pred_voxel_act_style,
        }
        self._wandb_summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.
        }
        if step_optimizer and self._lr_scheduler:
            self._scheduler.step()
            self._summaries["learning_rate"] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue = self._softmax_q_trans(q_trans[0])
        self._vis_max_coordinate = coords[0]
        self._vis_gt_coordinate = action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        q_trans_vis=True
        if step % self.cfg.framework.log_freq == 0  and rank == 0:
        # if step % 10 == 0 and rank == 0:
            r2bc_debug_print(f"{arm}_arm_predict: {self._vis_max_coordinate}")
            r2bc_debug_print(f"{arm}_gt: {self._vis_gt_coordinate}")
            rendered_img = visualise_voxel(
                voxel_grid[0].cpu().detach().numpy(),    # [10, 100, 100, 100]
                self._vis_translation_qvalue.detach().cpu().numpy() if q_trans_vis else None,
                self._vis_max_coordinate.detach().cpu().numpy(),
                self._vis_gt_coordinate.detach().cpu().numpy(),
                voxel_size=0.045,
                # voxel_size=0.1,   # more focus ??
                rotation_amount=np.deg2rad(-90),
                highlight_alpha=1.0,
                alpha=0.4,
            )
            os.makedirs('recon', exist_ok=True)
            # plot three images in one row with subplots:
            rgb_src = obs[0][0][0].squeeze(0).permute(1, 2, 0)  / 2 + 0.5

            fig, axs = plt.subplots(1, 4, figsize=(9, 3))
            # src
            axs[0].imshow(rgb_src.cpu().numpy())
            axs[0].title.set_text('src')

            axs[1].imshow(rendered_img)
            axs[1].text(0, 40, 'predicted', color='blue')
            axs[1].text(0, 80, 'gt', color='red')
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()

            if rank == 0:
                if self.cfg.framework.use_wandb:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    image = Image.open(buf)
                    wandb.log({"eval/recon_img": wandb.Image(image)}, step=step)

                    buf.close()
                    cprint(f'Saved to wandb', 'cyan')
                else:
                    plt.savefig(f'recon/{step}_rgb.png')
                    workdir = os.getcwd()
                    cprint(f'Saved {workdir}/recon/{step}_rgb.png locally', 'cyan')
        return {
            "total_loss": total_loss,
            "prev_layer_voxel_grid": prev_layer_voxel_grid,
            "prev_layer_bounds": prev_layer_bounds,
        }
    
    def update_wandb_summaries(self):
        summaries = dict()
        for k, v in self._wandb_summaries.items():
            summaries[k] = v
        return summaries
    
    def act(self, step: int, observation: dict, deterministic=False, arm=None) -> ActResult:
        if step == 0:
            r2bc_debug_print("[debug act qattention] arm:", arm)
            r2bc_debug_print(
                "[debug act qattention] low_dim keys:",
                [k for k in observation.keys() if "low_dim" in k]
            )
            if "low_dim_state" in observation:
                r2bc_debug_print(
                    "[debug act qattention] low_dim_state value:",
                    observation["low_dim_state"].detach().cpu().flatten().numpy()
                )
            
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get("prev_layer_voxel_grid", None)
        prev_layer_bounds = observation.get("prev_layer_bounds", None)
        lang_goal_tokens = observation.get("lang_goal_tokens", None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            (
                lang_goal_emb,
                lang_token_embs,
            ) = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio = None

        if self._include_low_dim_state:
            arm_low_dim_key = f"{arm}_low_dim_state" if arm in ("right", "left") else None

            if arm_low_dim_key is not None and arm_low_dim_key in observation:
                proprio_key = arm_low_dim_key
                proprio = observation[proprio_key]
            else:
                proprio_key = "low_dim_state"
                proprio = observation[proprio_key]

            if step == 0:
                r2bc_debug_print("[debug act qattention] proprio_key:", proprio_key)
                r2bc_debug_print("[debug act qattention] proprio shape:", tuple(proprio.shape))

            proprio = proprio[0].to(self._device)

        obs, pcd = self._act_preprocess_inputs(observation)

        if step == 0:
            r2bc_debug_print("[debug act qattention] after preprocess")
            r2bc_debug_print(
                "[debug act qattention] proprio after slice:",
                None if proprio is None else tuple(proprio.shape)
            )
            for n in self._camera_names:
                r2bc_debug_print(
                    "[debug act qattention]",
                    n,
                    "rgb raw", tuple(observation[f"{n}_rgb"].shape),
                    "pcd raw", tuple(observation[f"{n}_point_cloud"].shape),
                )

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        pcd = [p[0].to(self._device) for p in pcd]

        if step == 0:
            r2bc_debug_print("[debug act qattention] model input shapes")
            r2bc_debug_print(
                "[debug act qattention] proprio model:",
                None if proprio is None else tuple(proprio.shape)
            )
            for i, n in enumerate(self._camera_names):
                r2bc_debug_print(
                    "[debug act qattention]",
                    n,
                    "rgb model", tuple(obs[i][0].shape),
                    "pcd model", tuple(pcd[i].shape),
                )

        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = (
            prev_layer_voxel_grid.to(self._device)
            if prev_layer_voxel_grid is not None
            else None
        )
        prev_layer_bounds = (
            prev_layer_bounds.to(self._device)
            if prev_layer_bounds is not None
            else None
        )

        # inference
        q_trans, q_rot_grip, q_ignore_collisions, vox_grid, _, _ = self._q(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
            arm=arm,
        )

        # softmax Q predictions
        q_trans = self._softmax_q_trans(q_trans)
        q_rot_grip = (
            self._softmax_q_rot_grip(q_rot_grip)
            if q_rot_grip is not None
            else q_rot_grip
        )
        q_ignore_collisions = (
            self._softmax_ignore_collision(q_ignore_collisions)
            if q_ignore_collisions is not None
            else q_ignore_collisions
        )

        # argmax Q predictions
        (
            coords,
            rot_and_grip_indicies,
            ignore_collisions,
        ) = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = (
            ignore_collisions.int() if ignore_collisions is not None else None
        )

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2

        if arm == "left":
            r2bc_debug_print("[act trans debug] layer:", self._layer)
            r2bc_debug_print("[act trans debug] arm:", arm)
            r2bc_debug_print("[act trans debug] bounds:", bounds.detach().cpu().numpy())
            r2bc_debug_print("[act trans debug] res:", res.detach().cpu().numpy() if hasattr(res, "detach") else res)
            r2bc_debug_print("[act trans debug] coords:", coords.detach().cpu().numpy())
            r2bc_debug_print("[act trans debug] attention_coordinate:", attention_coordinate.detach().cpu().numpy())

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            "attention_coordinate": attention_coordinate,
            "prev_layer_voxel_grid": prev_layer_voxel_grid,
            "prev_layer_bounds": prev_layer_bounds,
        }
        info = {
            "voxel_grid_depth%d" % self._layer: vox_grid,
            "q_depth%d" % self._layer: q_trans,
            "voxel_idx_depth%d" % self._layer: coords,
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q_trans[0].detach()
        return ActResult(
            (coords, rot_grip_action, ignore_collisions_action),
            observation_elements=observation_elements,
            info=info,
        )

    def update_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary(
                "%s/update_qattention" % self._name,
                transforms.ToTensor()(
                    visualise_voxel(
                        self._vis_voxel_grid.detach().cpu().numpy(),
                        self._vis_translation_qvalue.detach().cpu().numpy(),
                        self._vis_max_coordinate.detach().cpu().numpy(),
                        self._vis_gt_coordinate.detach().cpu().numpy(),
                    )
                ),
            )
        ]

        for n, v in self._summaries.items():
            summaries.append(ScalarSummary("%s/%s" % (self._name, n), v))

        for name, crop in self._crop_summary:
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([ImageSummary("%s/crops/%s" % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            if param.grad is not None:
                summaries.append(
                    HistogramSummary("%s/gradient/%s" % (self._name, tag), param.grad)
                )
            summaries.append(
                HistogramSummary("%s/weight/%s" % (self._name, tag), param.data)
            )

        return summaries

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary(
                "%s/act_Qattention" % self._name,
                transforms.ToTensor()(
                    visualise_voxel(
                        self._act_voxel_grid.cpu().numpy(),
                        self._act_qvalues.cpu().numpy(),
                        self._act_max_coordinate.cpu().numpy(),
                    )
                ),
            )
        ]
    def concat_weights(self, param, target_size, dims=-1):
        """Repeat tensor along `dims` until that dimension reaches target_size.

        Used to adapt pure PerAct 128-dim weights to AnyBimanual 256-dim
        weights. The old implementation checked param.size(-1) even when
        concatenating along dim=0, which could accidentally create 512-dim
        tensors for decoder_cross_attn.to_out.
        """
        dim = dims
        if dim < 0:
            dim = param.dim() + dim

        while param.size(dim) < target_size:
            param = torch.cat([param, param], dim=dim)

        if param.size(dim) > target_size:
            slices = [slice(None)] * param.dim()
            slices[dim] = slice(0, target_size)
            param = param[tuple(slices)]

        return param

    def adapt_tensor_to_shape(self, param, target_shape):
        """Repeat checkpoint tensor along dimensions that need expansion.

        This is used for loading pure PerAct weights into the AnyBimanual
        architecture. Dimensions that already match the target are left
        unchanged.
        """
        if tuple(param.shape) == tuple(target_shape):
            return param

        if param.dim() != len(target_shape):
            return param

        out = param
        for dim, target_size in enumerate(target_shape):
            if out.size(dim) == target_size:
                continue

            if out.size(dim) < target_size:
                while out.size(dim) < target_size:
                    out = torch.cat([out, out], dim=dim)

                if out.size(dim) > target_size:
                    slices = [slice(None)] * out.dim()
                    slices[dim] = slice(0, target_size)
                    out = out[tuple(slices)]
            else:
                return param

        return out
    
    def load_weights(self, savedir: str):
        device = self._device
        weight_file = os.path.join(savedir, "%s.pt" % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        # Keep a copy of the current model state so shape-mismatched checkpoint
        # tensors can be safely skipped while preserving current initialization.
        current_state_dict = self._q.state_dict()
        merged_state_dict = dict(current_state_dict)
        qnet_is_ddp = isinstance(self._q._qnet, DDP)

        def to_model_key(k: str) -> str:
            """Convert checkpoint key to current model key.

            Checkpoints may have _qnet.module.* when saved with DDP.
            Current debug/training with ddp.num_devices=1 uses _qnet.*.
            """
            if qnet_is_ddp:
                if k.startswith("_qnet.") and not k.startswith("_qnet.module."):
                    return k.replace("_qnet.", "_qnet.module.", 1)
                return k
            else:
                if k.startswith("_qnet.module."):
                    return k.replace("_qnet.module.", "_qnet.", 1)
                return k

        module_prefix = "_qnet.module." if qnet_is_ddp else "_qnet."

        skip_anybimanual_modules = (
            os.environ.get("R2BC_RANDOM_INIT_ANYBIMANUAL", "0") == "1"
        )
        skipped_anybimanual_keys = 0
        skipped_shape_mismatch_keys = 0

        for raw_k, v in state_dict.items():
            # Voxelizer buffers depend on current batch size.
            # They are initialized in build(), so do not load them from checkpoints.
            if raw_k.startswith("_voxelizer"):
                continue

            k = to_model_key(raw_k)

            # Keep Skill Manager / Visual Aligner randomly initialized.
            # They were already created in create_agent(); skipping checkpoint keys
            # prevents pretrained AnyBimanual modules from overwriting them.
            if skip_anybimanual_modules and (
                "skill_manager" in k or "visual_aligner" in k
            ):
                skipped_anybimanual_keys += 1
                continue

            # pos encoding shape conversion for AnyBimanual
            if k == module_prefix + "pos_encoding":
                if (v.shape[1] != 8077 or v.shape[1] != 8154) and v.shape[1] < 154:
                    if self.anybimanual:
                        lang_max_seq_len = 154
                    else:
                        lang_max_seq_len = 77

                    spatial_size = v.shape[1]
                    input_dim_before_seq = v.shape[-1]
                    flattened_v = v.view(1, -1, input_dim_before_seq)

                    new_pos_encoding = torch.randn(
                        1,
                        lang_max_seq_len,
                        input_dim_before_seq,
                        device=device,
                    )

                    merged_pos_encoding = torch.cat(
                        [flattened_v, new_pos_encoding],
                        dim=1,
                    )

                    merged_state_dict[module_prefix + "pos_encoding"] = merged_pos_encoding
                else:
                    merged_state_dict[module_prefix + "pos_encoding"] = v

            elif k.startswith(module_prefix + "cross_attend_blocks"):
                if self.anybimanual and v.size(-1) == 128:
                    merged_state_dict[k] = self.concat_weights(v, 256)
                elif k in merged_state_dict:
                    merged_state_dict[k] = v

            elif k.startswith(module_prefix + "decoder_cross_attn"):
                if self.anybimanual and k in merged_state_dict:
                    merged_state_dict[k] = self.adapt_tensor_to_shape(
                        v, current_state_dict[k].shape
                    )
                elif k in merged_state_dict:
                    merged_state_dict[k] = v

            elif k == module_prefix + "up0.conv_up.0.conv3d.weight":
                if self.anybimanual and v.size(1) == 128:
                    merged_state_dict[k] = self.concat_weights(v, 256, 1)
                elif k in merged_state_dict:
                    merged_state_dict[k] = v

            elif k.startswith(module_prefix + "dense0"):
                if self.anybimanual and v.size(-1) == 1024:
                    merged_state_dict[k] = torch.cat([v, v[:, :512]], dim=-1)
                elif k in merged_state_dict:
                    merged_state_dict[k] = v

            elif k in merged_state_dict:
                merged_state_dict[k] = v

            else:
                logging.warning("key %s not found in checkpoint", k)

        if skip_anybimanual_modules:
            r2bc_debug_print(
                "[load_weights] R2BC_RANDOM_INIT_ANYBIMANUAL=1: "
                f"skipped {skipped_anybimanual_keys} "
                "skill_manager/visual_aligner checkpoint keys"
            )

        # Some PerAct checkpoints have slightly different architecture dimensions.
        # If a checkpoint tensor shape does not match the current model tensor shape,
        # keep the current model initialization for that key instead of crashing.
        for k in list(merged_state_dict.keys()):
            if k not in current_state_dict:
                continue

            if merged_state_dict[k].shape != current_state_dict[k].shape:
                r2bc_debug_print(
                    "[load_weights] skip shape mismatch:",
                    k,
                    "ckpt", tuple(merged_state_dict[k].shape),
                    "model", tuple(current_state_dict[k].shape),
                )
                merged_state_dict[k] = current_state_dict[k]
                skipped_shape_mismatch_keys += 1

        if skipped_shape_mismatch_keys > 0:
            r2bc_debug_print(
                "[load_weights] skipped "
                f"{skipped_shape_mismatch_keys} shape-mismatch checkpoint keys"
            )

        self._q.load_state_dict(merged_state_dict)
        r2bc_debug_print("loaded weights from %s" % weight_file)


    def save_weights(self, savedir: str):
        torch.save(self._q.state_dict(), os.path.join(savedir, "%s.pt" % self._name))