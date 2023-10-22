### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

# Gym
import gym

# SB3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Simulation (pygame, pymunk)
import pygame
import pymunk

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData
from CustomRacing2D_env import CustomRacing2DEnv, define_boundaries, boundaries, inPoly, playNEpisodes

if __name__ == "__main__":
    # Load points
    # Pull waypoints from .gpx file
    points = removeDuplicatePoints(read_gpx('../gpx/windy_road.gpx', 1))
    points = points[100:200]
    points = scaleData(points)
    points = points * 900 + 50
    points = points[:, :2]

    # For some number of iterations,
    # Add more points in between each set of points
    for i in range(2):
        new_points = []
        for i in range(len(points) - 1):
            new_points.append(points[i])
            new_points.append((points[i] + points[i + 1]) / 2)
        new_points.append(points[-1])
        points = np.array(new_points)

    # min_vector_length = min([np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)]) * 2

    # Create boundaries
    boundary_points = define_boundaries(points, 50)

    # Create environment
    env = CustomRacing2DEnv(points, boundary_points)

    # Load agent model
    # model = PPO.load("ppo_custom_racing_2d")
    model = PPO.load("../models/temp_model")

    # # Train the agent on the new environment
    # model.set_env(env)
    # model.learn(total_timesteps=300000)

    # # Save the agent
    # model.save("../models/ppo_custom_racing_2d_exp_2")

    # Evaluate agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

    # Play episodes
    playNEpisodes(1, env, model)