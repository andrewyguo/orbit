from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from abc import abstractmethod
from dataclasses import MISSING
from typing import Sequence, Tuple

import omni
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.cloner import GridCloner
from pxr import PhysxSchema

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation, ArticulationCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.sim import SimulationCfg, SimulationContext
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import sample_uniform
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

from omni.isaac.orbit_tasks.classic.cartpole.cartpole_env_2 import OIGEEnv, OIGEBaseCfg

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@configclass
class AntEnvCfg(OIGEBaseCfg):
    power_scale: float = 0.5
    heading_weight: float = 0.5
    up_weight: float = 0.1 

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5

    death_cost: float = -2.0
    termination_height: float = 0.31 

    angular_velocity_scale: float = 1.0
    dol_vel_scale: float = 0.1
    contact_force_scale: float = 0.1


class AntEnv(OIGEEnv):
    def __init__(self, cfg: AntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone() * self.cfg.power_scale * self.joint_gears

        indices = torch.arange(self.robot.count, dtype=torch.int32, device=self.sim.device) 

        self.robot.set_joint_effort_target(self.actions, indices)

    def _post_physics_step(self):
        self.torso_position, self.torso_rotation = self.robot.get_world_poses(clone=False) # NOTE: get_world_poses might not be defined in the class for ant, need to define potentially. Thinking of a question: where are the actual robots placed in the scene? 
        
        velocities = self.robot.get_velocities(clone=False) # NOTE: Need to make sure this is how to get velocity 
        self.velocity = velocities[:, 0:3]
        self.ang_velocity = velocities[:, 3:6]

        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel

        to_target = self.targets - self.torso_position
        to_target[:, 2] = 0.0

        self.torso_quat, self.up_proj, self.heading_proj, self.up_vec, self.heading_vec = compute_heading_and_up(
            self.torso_rotation, self.inv_start_rot, to_target, self.basis_vec0, self.basis_vec1, 2
        )

        self.vel_loc, self.angvel_loc, self.roll, self.pitch, self.yaw, self.angle_to_target = compute_rot(
            self.torso_quat, self.velocity, self.ang_velocity, self.targets, self.torso_position
        )

        self.sensor_force_torques = self.robot.get_measured_joint_forces(joint_indices=self._sensor_indices) # TODO: make sure get_measured_joint_forces is defined 
        # ALSO TODO: define _sensor_indices

        self.dof_pos_scaled = torch_utils.maths.unscale(self.dof_pos, self.dof_limits_lower, self.dof_limits_upper)

        # define torso position, rotation, ect.  

    def _get_observations(self) -> dict:
        
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.sensor_force_torques.reshape(self.num_envs, -1) * self.cfg.contact_force_scale,
                self.actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    def _get_reward(self) -> torch.Tensor:
        # obs_buf = self._get_observations()["policy"]
        
        heading_weight_tensor = torch.ones_like(self.heading_proj) * self.cfg.heading_weight
        heading_reward = torch.where(self.heading_proj > 0.8, heading_weight_tensor, self.cfg.heading_weight * self.heading_proj / 0.8)

        # aligning up axis of robot and environment
        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(self.up_proj > 0.93, up_reward + self.cfg.up_weight, up_reward)

        # energy penalty for movement
        actions_cost = torch.sum(self.actions**2, dim=-1)
        # TODO: define robot.num_dof 
        electricity_cost = torch.sum( 
            torch.abs(self.actions * self.dof_vel * self.cfg.dof_vel_scale) * self.motor_effort_ratio.unsqueeze(0), dim=-1
        ) # NOTE: might not be correct, replaced obs_buf[:, 12 ...] with this 
 
        # reward for duration of staying alive
        alive_reward = torch.ones_like(self.potentials) * self.cfg.alive_reward_scale
        progress_reward = self.potentials - self.prev_potentials

        total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - self.cfg.actions_cost_scale * actions_cost
            - self.cfg.energy_cost_scale * electricity_cost
            # - dof_at_limit_cost # TODO: add this back if needed 
        )

        # adjust reward for fallen agents
        total_reward = torch.where(
            self.torso_position[:, 2].view(-1, 1) < self.cfg.termination_height, torch.ones_like(total_reward) * self.cfg.death_cost, total_reward
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        num_resets = len(env_ids)
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # randomize DOF positions and velocities
             # TODO: Define robot.num_dof or use something else 
        dof_pos = torch_utils.torch_rand_float(-0.2, 0.2, (num_resets, self.robot.num_dof), device=self.sim.device)
        dof_pos[:] = torch_utils.tensor_clamp(self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper)
        dof_vel = torch_utils.torch_rand_float(-0.1, 0.1, (num_resets, self.robot.num_dof), device=self.sim.device)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self.sim.device)

        # apply resets
            # TODO: define set_joint_positions, set_joint_velocities, set_world_poses, set_velocities
        self.robot.set_joint_positions(dof_pos, indices=env_ids)
        self.robot.set_joint_velocities(dof_vel, indices=env_ids)

        self.robot.set_world_poses(root_pos, root_rot, indices=env_ids)
        self.robot.set_velocities(root_vel, indices=env_ids)


        to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

    def _post_physics_step(self):
        pass

    def _post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self.robot.get_world_poses()
        self.initial_dof_pos = self.robot.data.joint_pos

        # from ant task 
        self.joint_gears = torch.tensor([15, 15, 15, 15, 15, 15, 15, 15], dtype=torch.float32, device=self.sim.device)
        dof_limits = self.robot.get_dof_limits() # define get_dof_limits
        self.dof_limits_lower = dof_limits[0, :, 0].to(self.sim.device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)

        # from locomotion task 
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self.sim.device).repeat(
            self.num_envs
        )
        self.prev_potentials = self.potentials.clone()

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()




