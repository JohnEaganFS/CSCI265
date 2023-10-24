### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from typing import Callable

# Gym
import gym

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Simulation (pygame, pymunk)
import pygame
import pymunk

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData

### Misc. Functions ###
def initialize_pygame(width, height):
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("Racing Environment")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    return screen, clock

### Global Variables ###
# Model parameters
max_steps = 1000
total_timesteps = 1000000

# Pygame parameters
FPS = 120
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

# Simulation parameters


### Observation Space ###

### Action Space ###

### Environment ###




### Main ###
if __name__ == "__main__":
    # Initialize pygame
    screen, clock = initialize_pygame(1000, 1000)


    print("Hello, world!")