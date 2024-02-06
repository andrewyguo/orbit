# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from omni.isaac.orbit_assets.cartpole import CARTPOLE_CFG
from pxr import PhysxSchema

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation, ArticulationCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.sim import SimulationCfg, SimulationContext
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import sample_uniform


@configclass
class OIGEBaseCfg:
    # simulation
    viewer: ViewerCfg = ViewerCfg(eye=(8.0, 0.0, 5.0))
    sim: SimulationCfg = SimulationCfg(dt=1 / 60)
    scene = None
    # env
    num_envs: int = 4096
    max_episode_length: float = MISSING
    env_spacing: float = MISSING
    control_frequency_inv: int = MISSING
    num_actions: int = MISSING
    num_observations: int = MISSING
    add_default_lights: bool = True
    # robot
    robot_cfg: ArticulationCfg = MISSING


class OIGEEnv(gym.Env):
    cfg: OIGEBaseCfg

    def __init__(self, cfg: OIGEBaseCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.decimation = cfg.control_frequency_inv
        self.num_actions = cfg.num_actions
        self.num_observations = cfg.num_observations

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        # self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # simulation
        self.sim = SimulationContext(self.cfg.sim)
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)

        self.stage = omni.usd.get_context().get_stage()

        self.cloner = GridCloner(spacing=cfg.env_spacing)
        self.cloner.define_base_env("/World/envs")
        self.env_prim_paths = self.cloner.generate_paths("/World/envs/env", self.num_envs)
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        env_origins = self.cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=False,
            copy_from_source=True,
        )
        self.env_origins = torch.tensor(env_origins, device=self.sim.device, dtype=torch.float32)

        self._setup_scene()

        self.cloner.replicate_physics(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            base_env_path="/World/envs",
            root_path="/World/envs/env_",
        )

        # disable collisions between the different environments
        physics_scene_prim_path = None
        for prim in self.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                physics_scene_prim_path = prim.GetPrimPath()
                break
        self.cloner.filter_collisions(
            physics_scene_prim_path,
            "/World/collisions",
            self.env_prim_paths,
            global_paths=[],  # NOTE: Once we have envs that walk on the ground, global path will have to be set see self._global_prim_paths in interactive_scene.py
        )

        # the reset here will call the initialization callback of the articulation
        self.sim.reset()

        # RL specifics
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.long)
        self.max_episode_length = math.ceil(self.cfg.max_episode_length / (self.cfg.sim.dt * self.decimation))
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    @abstractmethod
    def _pre_physics_step(self):
        return NotImplementedError

    @abstractmethod
    def _get_observations(self) -> torch.Tensor:
        return NotImplementedError

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        return NotImplementedError

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return NotImplementedError

    def step(self, actions: torch.Tensor):
        self._pre_physics_step(actions)

        for _ in range(self.decimation):
            # set actions into simulator and step the actuator network if there is one
            self.robot.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.robot.update(dt=self.cfg.sim.dt)
        # perform rendering if gui is enabled
        if self.sim.has_gui():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)

        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reward_buf = self._get_rewards()

        reset_buf = self.reset_terminated | self.reset_time_outs
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self._reset_idx(reset_env_ids)

        self.obs_buf = self._get_observations()
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, {}

    def reset(self):
        self._reset_idx(None)
        return self._get_observations(), {}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # reset the robots
        self.robot.reset(env_ids)
        # reset the buffers
        self.episode_length_buf[env_ids] = 0.0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        if self.cfg.add_default_lights:
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)


@configclass
class CartpoleEnvCfg(OIGEBaseCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # env
    max_episode_length = 5.0
    env_spacing = 4.0
    control_frequency_inv = 2
    action_scale = 100.0 # [N]
    num_actions = 1
    num_observations = 4
    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25] # the range in which the pole angle is sampled from on reset [rad]
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class CartpoleEnv(OIGEEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        self.robot.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        rew_alive = self.cfg.rew_scale_alive
        rew_termination = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_pole_pos = self.cfg.rew_scale_pole_pos * torch.sum(
            torch.square(self.robot.data.joint_pos[:, self._pole_dof_idx]), dim=1
        )
        rew_cart_vel = self.cfg.rew_scale_cart_vel * torch.sum(
            torch.abs(self.robot.data.joint_vel[:, self._cart_dof_idx]), dim=1
        )
        rew_pole_vel = self.cfg.rew_scale_pole_vel * torch.sum(
            torch.abs(self.robot.data.joint_vel[:, self._pole_dof_idx]), dim=1
        )
        total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.robot.data.joint_pos[:, self._pole_dof_idx] > self.cfg.max_cart_pos), dim=1
        )
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

