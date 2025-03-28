# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sensors import ContactSensor

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
    # skate_rot_euler = torch.cat([skate_rot_euler[0].unsqueeze(1), skate_rot_euler[1].unsqueeze(1), skate_rot_euler[2].unsqueeze(1)], dim=1)
    return skate_rot_euler[2].unsqueeze(1)

def skate_feet_contact_obs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward for feet contact with skateboard"""
    FR_contact_sensor: ContactSensor = env.scene.sensors["FR_contact"]
    FL_contact_sensor: ContactSensor = env.scene.sensors["FL_contact"]
    RR_contact_sensor: ContactSensor = env.scene.sensors["RR_contact"]
    RL_contact_sensor: ContactSensor = env.scene.sensors["RL_contact"]
    FR_contact_sensor.data.force_matrix_w.squeeze(1)
    cat = torch.cat([FR_contact_sensor.data.force_matrix_w.squeeze(1), FL_contact_sensor.data.force_matrix_w.squeeze(1),
     RR_contact_sensor.data.force_matrix_w.squeeze(1), RL_contact_sensor.data.force_matrix_w.squeeze(1)], dim=1)
    # heigh_mask = torch.cat([FR_contact_sensor.data.pos_w[..., 2], FL_contact_sensor.data.pos_w[..., 2],
    #  RR_contact_sensor.data.pos_w[..., 2], RL_contact_sensor.data.pos_w[..., 2]], dim=1)
    
    contact_tensor = torch.any(cat != 0, dim=2).float()
    return contact_tensor 




# def skate_feet_positions(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     skate_asset_cfg = SceneEntityCfg,
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     skate_asset: RigidObject = env.scene[skate_asset_cfg.name]
#     foot_pose = asset.data.body_pos_w[:, asset_cfg.body_ids]
#     skate_pose = skate_asset.data.root_pos_w.unsqueeze(1)
#     skate_pose = skate_pose.repeat(1, 4, 1)
#     target_pose = torch.tensor([[0.2, 0.15, 0.1], [0.2, -0.15, 0.1], [-0.2, 0.15, 0.1], [-0.2, -0.15, 0.1]], device=env.device) + skate_pose
#     relative_pos = target_pose - foot_pose
#     relative_pos = relative_pos.view(*relative_pos.shape[:-2], -1)
#     return relative_pos

# def target_vel(
#     env: ManagerBasedRLEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
#     skate_asset_cfg: SceneEntityCfg = SceneEntityCfg("skateboard")
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot_asset: RigidObject = env.scene[robot_asset_cfg.name]
#     skate_asset: RigidObject = env.scene[skate_asset_cfg.name]

#     target_vel = skate_asset.data.root_pos_w[:, :2] - robot_asset.data.root_pos_w[:, :2]
#     norm_v = torch.norm(target_vel, dim=1, keepdim=True)
#     norm_v = norm_v.repeat(1, 2)
#     vector_norm = 0.5
#     unit_vector = target_vel / norm_v

#     # Replace elements in tensor1 with corresponding elements from tensor3 based on the condition
#     target_vel[norm_v > vector_norm] = unit_vector[norm_v > vector_norm] * vector_norm

#     return target_vel
