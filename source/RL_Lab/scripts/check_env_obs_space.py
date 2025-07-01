#!/usr/bin/env python3

import isaaclab
import gymnasium as gym
import numpy as np

def main():
    """Check the observation space of an environment."""
    # Create the environment
    env_id = "Custom-Factory-PegInsert-RGB-Camera-Direct-v0"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, headless=False)
    
    # Reset the environment and get the initial observation
    obs, _ = env.reset()
    
    # Print observation space information
    print("\nObservation Space Information:")
    print(f"Observation space: {env.observation_space}")
    
    if isinstance(obs, dict):
        print("\nObservation is a dictionary with keys:")
        for key, value in obs.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if hasattr(value, 'min') and hasattr(value, 'max'):
                print(f"    min={value.min()}, max={value.max()}")
    else:
        print(f"\nObservation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Observation min: {obs.min()}")
        print(f"Observation max: {obs.max()}")
    
    # Print action space information
    print("\nAction Space Information:")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()