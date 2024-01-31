# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test a task with zero actions.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from omni.isaac.orbit_tasks.classic.cartpole.cartpole_env import CartpoleEnvCfg, CartpoleEnv

def main():
    cartpole_cfg = CartpoleEnvCfg()
    env = CartpoleEnv(cartpole_cfg)
    env.reset()
    actions = torch.zeros_like(env.actions)
    actions[:] = 0.5
    while simulation_app.is_running():
        observations, rewards, is_terminated, is_timed_out, extras = env.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
