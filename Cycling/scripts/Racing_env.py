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
import pygame.gfxdraw
import pymunk
import pickle

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData
from CustomRacing2D_env import draw_waypoints, getNewHeading

### Misc. Functions ###
def initialize_pygame(width, height):
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("Racing Environment")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    return screen, clock

def draw_walls(screen, pl_set):
    for pl in pl_set:
        # Draw filled polygon (each pl is a list of points)
        pygame.draw.polygon(screen, (255, 0, 0), pl)

def getNewSpeed(speed, throttle, speed_limit):
    force = throttle

    force += np.random.normal(0, 0.05)

    new_speed = speed + force

    if new_speed > speed_limit:
        new_speed = speed_limit
    elif new_speed < 0:
        new_speed = 0
    
    return new_speed

### Global Variables ###
# Model parameters
max_steps = 1000
total_timesteps = 1000000

# Pygame parameters
FPS = 120

# Simulation parameters


### Observation Space ###
# Observation space is a 2D array of pixel values (image around the car)
def defineObservationSpace():
    return gym.spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)

### Action Space ###
# Action space is steering and throttle
actions = ['steer', 'throttle']
def defineActionSpace():
    lows = {
        'steer': -1,
        'throttle': -1
    }
    highs = {
        'steer': 1,
        'throttle': 1
    }
    low = np.array([lows[act] for act in actions])
    high = np.array([highs[act] for act in actions])
    return gym.spaces.Box(low, high, dtype=np.float32)

### Environment ###
class RacingEnv(gym.Env):
    def __init__(self, map):
        # Load map data
        with open(map, 'rb') as f:
            self.static_space, self.points, self.boundaries, self.screen_size, self.walls = pickle.load(f)
            self.points = self.points[1:]

        # Initialize pygame
        self.screen, self.clock = initialize_pygame(self.screen_size[0], self.screen_size[1])
    
        # Initialize environment variables
        self.observation_space = defineObservationSpace()
        self.action_space = defineActionSpace()
        # self.reward_range = (-np.inf, np.inf)

        self.render()



    def observation(self):
        # Get observation (100x100 image around the car)
        pass

    def reset(self):
        # Reset environment
        space = self.static_space
        # Add the car   
        car = pymunk.Body(10, 1, body_type=pymunk.Body.DYNAMIC)
        car.position = (self.points[0][0], self.points[0][1])
        car_shape = pymunk.Circle(car, 2)
        car_shape.elasticity = 1
        space.add(car, car_shape)
        

        self.state = {
            'heading': np.random.uniform(-np.pi, np.pi)
        }

        self.speed_limit = 200

    def step(self, action):
        # Take action
        steer, throttle = action

        # Get new heading
        new_heading = getNewHeading(self.state['heading'], steer)
        self.state['heading'] = new_heading
        # Get new speed


    def render(self):

        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True

            # Draw waypoints
            draw_waypoints(self.screen, self.points, 0, 1)
            # Draw walls
            draw_walls(self.screen, self.walls)

            # Update screen
            pygame.display.flip()
            self.clock.tick(FPS)

    def close(self):
        pass




### Main ###
if __name__ == "__main__":

    # Initialize environment
    env = RacingEnv("../maps/map_10_30_800_800.pkl")



    print("Hello, world!")