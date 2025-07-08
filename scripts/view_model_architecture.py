#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to view RL-Games model architecture before training."""

import argparse
import os
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="View RL-Games model architecture.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--config", type=str, required=True, help="Path to RL-Games config file.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # enable cameras for RGBD observations

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import RL_Lab.tasks  # noqa: F401

def main():
    """View model architecture."""
    # Load the RL-Games config
    with open(args_cli.config, 'r') as f:
        agent_cfg = yaml.safe_load(f)
    
    # Create environment to get observation and action spaces
    env = gym.make(args_cli.task)
    
    # Get observation and action spaces
    obs_space = env.observation_space
    act_space = env.action_space
    
    print("\n===== Environment Information =====")
    print(f"Observation Space: {obs_space}")
    print(f"Action Space: {act_space}")
    
    # Wrap for rl-games
    rl_device = "cuda:0"
    env = RlGamesVecEnvWrapper(env, rl_device, math.inf, math.inf)
    
    # Register the environment
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    # Extract network configuration
    network_config = agent_cfg["params"]["network"]
    
    print("\n===== Network Configuration =====")
    print_dict(network_config)
    
    # Create the model manually to inspect
    print("\n===== Creating Model =====")
    
    # Initialize the network builder
    builder = torch_ext.ModelBuilder()
    
    # Get the network parameters from config
    network_params = agent_cfg["params"]["network"]
    
    # Create input shape based on environment observation space
    if isinstance(obs_space, gym.spaces.Dict):
        # For dict observation spaces (like with policy and critic)
        if "policy" in obs_space.spaces:
            input_shape = obs_space.spaces["policy"].shape
        else:
            input_shape = list(obs_space.spaces.values())[0].shape
    else:
        input_shape = obs_space.shape
    
    print(f"Input shape: {input_shape}")
    
    # Get action space size
    action_space_size = act_space.shape[0] if hasattr(act_space, "shape") else act_space.n
    
    # Create a dummy network to inspect
    try:
        # For continuous action spaces
        if isinstance(act_space, gym.spaces.Box):
            model = ModelA2CContinuousLogStd(input_shape, action_space_size, builder, **network_params)
            print("\n===== Model Architecture =====")
            
            # Print CNN architecture if present
            if hasattr(model.actor, 'cnn') and model.actor.cnn is not None:
                print("\nActor CNN:")
                print(model.actor.cnn)
            
            # Print actor MLP
            print("\nActor MLP:")
            print(model.actor.mlp)
            
            # Print mu and sigma layers
            print("\nActor Output Layers:")
            print(f"mu: {model.mu}")
            print(f"sigma: {model.sigma}")
            
            # Print critic architecture
            print("\nCritic Architecture:")
            if hasattr(model.critic, 'cnn') and model.critic.cnn is not None:
                print("\nCritic CNN:")
                print(model.critic.cnn)
            
            print("\nCritic MLP:")
            print(model.critic.mlp)
            
            # Print value layer
            print("\nCritic Output Layer:")
            print(f"Value: {model.value}")
            
            # Print total parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nTotal parameters: {total_params:,}")
            
            # Try to print a forward pass with dummy data
            print("\n===== Forward Pass Shape =====")
            dummy_input = torch.zeros((1, *input_shape), device=rl_device)
            try:
                with torch.no_grad():
                    action, value, entropy = model({"obs": dummy_input})
                    print(f"Action output shape: {action.shape}")
                    print(f"Value output shape: {value.shape}")
            except Exception as e:
                print(f"Could not perform forward pass: {e}")
        else:
            print("Only continuous action spaces are supported for visualization")
    except Exception as e:
        print(f"Error creating model: {e}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    import math  # Required for math.inf
    # run the main function
    main()
    # close sim app
    simulation_app.close()