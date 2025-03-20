# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def skate_pos_rel(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    skate_pos_rel = env.scene["skate_transform"].data.target_pos_source.squeeze(1)
    return skate_pos_rel

def skate_rot_rel(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    skate_rot_quat = env.scene["skate_transform"].data.target_quat_source.squeeze(1)
    skate_rot_euler = euler_xyz_from_quat(skate_rot_quat)
    skate_rot_euler = torch.cat([skate_rot_euler[0].unsqueeze(1), skate_rot_euler[1].unsqueeze(1), skate_rot_euler[2].unsqueeze(1)], dim=1)
    return skate_rot_euler

def target_vel(
    env: ManagerBasedRLEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    skate_asset_cfg: SceneEntityCfg = SceneEntityCfg("skateboard")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: RigidObject = env.scene[robot_asset_cfg.name]
    skate_asset: RigidObject = env.scene[skate_asset_cfg.name]

    target_vel = skate_asset.data.root_pos_w[:, :2] - robot_asset.data.root_pos_w[:, :2]
    norm_v = torch.norm(target_vel, dim=1, keepdim=True)
    norm_v = norm_v.repeat(1, 2)
    vector_norm = 1
    unit_vector = target_vel / norm_v

    # Replace elements in tensor1 with corresponding elements from tensor3 based on the condition
    target_vel[norm_v > vector_norm] = unit_vector[norm_v > vector_norm] * vector_norm

    return target_vel


# def skate_pos_rel(
#     env: ManagerBasedEnv,
#     robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     skate_asset_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
# ) -> torch.Tensor:
#     """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
#     # extract the used quantities (to enable type-hinting)
#     robot_asset: RigidObject = env.scene[robot_asset_cfg.name]
#     skate_asset: RigidObject = env.scene[skate_asset_cfg.name]
#     skate_pos_rel = skate_asset.data.root_pos_w - robot_asset.data.root_pos_w

#     return skate_pos_rel

# def skate_rot_rel(
#     env: ManagerBasedEnv,
#     robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     skate_asset_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
# ) -> torch.Tensor:
#     """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
#     # extract the used quantities (to enable type-hinting)
#     robot_asset: RigidObject = env.scene[robot_asset_cfg.name]
#     skate_asset: RigidObject = env.scene[skate_asset_cfg.name]

#     skate_q = skate_asset.data.root_quat_w
#     robot_q = robot_asset.data.root_quat_w
#     robot_q_inv = torch.tensor([robot_q[0], -robot_q[1], -robot_q[2], -robot_q[3]])
#     skate_rot_rel = quaternion_multiply(skate_q, robot_q_inv)

#     return skate_rot_rel

# def quaternion_multiply(quaternion1, quaternion0):
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return torch.tensor([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=torch.float64)