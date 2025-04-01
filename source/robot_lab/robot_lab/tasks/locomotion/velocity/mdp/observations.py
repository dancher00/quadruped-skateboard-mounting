# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul, quat_conjugate, quat_rotate_inverse
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
    """Position of skateboard in robot's frame"""
    skate_pos_rel = env.scene["skate_transform"].data.target_pos_source.squeeze(1)
    return skate_pos_rel

def skate_rot_rel(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Orientation (yaw) of skateboard in robot's frame"""

    skate_rot_quat = env.scene["skate_transform"].data.target_quat_source.squeeze(1)
    skate_rot_euler = euler_xyz_from_quat(skate_rot_quat)
    # skate_rot_euler = torch.cat([skate_rot_euler[0].unsqueeze(1), skate_rot_euler[1].unsqueeze(1), skate_rot_euler[2].unsqueeze(1)], dim=1)
    return skate_rot_euler[2].unsqueeze(1)

def skate_feet_contact_obs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Returns the presence of contact with the skate for each foot"""

    FR_contact_sensor: ContactSensor = env.scene.sensors["FR_contact"]
    FL_contact_sensor: ContactSensor = env.scene.sensors["FL_contact"]
    RR_contact_sensor: ContactSensor = env.scene.sensors["RR_contact"]
    RL_contact_sensor: ContactSensor = env.scene.sensors["RL_contact"]
    FR_contact_sensor.data.force_matrix_w.squeeze(1)
    cat = torch.cat([FR_contact_sensor.data.force_matrix_w.squeeze(1), FL_contact_sensor.data.force_matrix_w.squeeze(1),
     RR_contact_sensor.data.force_matrix_w.squeeze(1), RL_contact_sensor.data.force_matrix_w.squeeze(1)], dim=1)
    
    contact_tensor = torch.any(cat != 0, dim=2).float()
    return contact_tensor 

def feet_pose(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Position of each foot in robot's frame"""

    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
    return footpos_in_body_frame.view(env.num_envs, -1)


def quat_conjugate(quaternions):
    """Compute the conjugate of a quaternion."""
    return quaternions[..., 0:1] * torch.tensor([1, -1, -1, -1], device=quaternions.device)

def quat_mul(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), dim=-1)

def rotate_vectors_by_quaternion(vectors, quaternions):
    """Rotate vectors by quaternions.
    
    Args:
        vectors: Shape is (M, 3)
        quaternions: Shape is (N, 4)

    Returns:
        Tensor of vectors rotated by quaternions. Shape is (N, M, 3)
    """
    N = quaternions.shape[0]
    M = vectors.shape[0]

    # Create a tensor of shape (N, M, 4) for vectors in quaternion form
    v_quat = torch.zeros((N, M, 4), device=vectors.device)
    v_quat[..., 1:] = vectors.unsqueeze(0)  # Assign vectors to the last 3 dimensions

    # Expand quaternions to match the shape of vectors
    quaternions_expanded = quaternions.unsqueeze(1)  # Shape: (N, 1, 4)
    q_conjugate = quat_conjugate(quaternions_expanded)

    # Rotate: q * v * q_conjugate
    rotated_vectors = quat_mul(quat_mul(quaternions_expanded, v_quat), q_conjugate)

    return rotated_vectors[..., 1:]  # Return only the vector part

def transform_vectors_to_parent_frame(target_positions, target_quaternions, vectors):
    """Transform vectors from target frames to parent frame using batch processing.
    
    Args:
        target_positions: Shape is (N, 3)
        target_quaternions: Shape is (N, 4)
        vectors: Shape is (M, 3)

    Returns:
        Tensor of vectors rotated by target quaternions and translated by target vectors. Shape is (N, M, 3)
    """
    # Rotate the vectors using quaternions
    rotated_vectors = rotate_vectors_by_quaternion(vectors, target_quaternions)  # Shape: (N, M, 3)

    # Add the target positions to the rotated vectors
    result_tensor = rotated_vectors + target_positions.unsqueeze(1)  # Broadcasting: (N, M, 3)

    return result_tensor.reshape(result_tensor.shape[0], -1)  # Shape: (N, M*3)

def rectangle_perimeter_tensor(width, height, interval):
    """Generate a tensor with coordinates of all points on the perimeter of a rectangle.
    
    Args:
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        interval (float): The interval between points on the perimeter.
    
    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing the coordinates of points on the perimeter.
    """
    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2
    
    # Initialize a list to hold the points
    points = []

    # Bottom edge (from (-half_width, -half_height) to (half_width, -half_height))
    for x in torch.arange(-half_width, half_width + interval, interval):
        points.append(torch.tensor([x, -half_height, 0.1]))

    # Right edge (from (half_width, -half_height) to (half_width, half_height))
    for y in torch.arange(-half_height + interval, half_height + interval, interval):
        points.append(torch.tensor([half_width, y, 0.1]))

    # Top edge (from (half_width, half_height) to (-half_width, half_height))
    for x in torch.arange(half_width, -half_width - interval, -interval):
        points.append(torch.tensor([x, half_height, 0.1]))

    # Left edge (from (-half_width, half_height) to (-half_width, -half_height))
    for y in torch.arange(half_height - interval, -half_height - interval, -interval):
        points.append(torch.tensor([-half_width, y, 0.1]))

    # Convert list of points to a tensor
    perimeter_tensor = torch.stack(points)

    return perimeter_tensor

def skate_point_cloud(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Positions of points along the skateboard’s edge in robot's frame, spaced with "interval" """
    # extract the used quantities (to enable type-hinting)
    skate_pos = env.scene["skate_transform"].data.target_pos_source.squeeze(1)
    skate_rot_quat = env.scene["skate_transform"].data.target_quat_source.squeeze(1)

    # vectors = rectangle_perimeter_tensor(0.575, 0.43, 0.1).to(env.device)
    length = 0.575
    width = 0.25
    interval = 0.083
    vectors = rectangle_perimeter_tensor(length, width, interval).to(env.device)
    vectors_transformed = transform_vectors_to_parent_frame(skate_pos, skate_rot_quat, vectors)

    return vectors_transformed
