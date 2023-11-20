# Ideas:
# 1. No map rotation, just with arrow tip
# 2. Energy limitation, coop by being close to other car (use less energy)

### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from typing import Callable
import time
from cProfile import Profile

# PyTorch
import torch as th
import torch.nn as nn

# Gym
import gym
from gym import spaces

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, BaseCallback

# Simulation (pygame, pymunk)
import pygame
import pymunk
import pickle

# Custom (other scripts)
from misc_functions import *
from read_gpx import read_gpx, removeDuplicatePoints, scaleData
from RacingMaps_env import RacingEnv as RacingEnvMaps

### Global Variables ###
# Model parameters
max_steps = 2000
total_timesteps = 1000000
observation_size = 64
num_agents = 2
n_envs = 1

hyperparameters = {
    'n_steps': 4096,
    'batch_size': 1024,
    'gamma': 0.99,
    'vf_coef': 0.5,
    # 'learning_rate': 0.0001,
    'gae_lambda': 0.95,
    'clip_range': 0.2
}

# Pygame parameters
FPS = 120

### Observation Space ###
# Observation space is a 2D array of pixel values (image around the car)
def defineObservationSpace():
    return gym.spaces.Box(low=0, high=255, shape=(3, observation_size, observation_size), dtype=np.uint8)

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
    def __init__(self, maps, max_steps, num_agents, model=None, evaluating=False, model_filename=None):
        # Initialize environment variables (should not change)
        self.maps = maps
        self.max_steps = max_steps
        self.current_map = None
        self.num_agents = num_agents
        self.other_agent = model
        self.observation_space = defineObservationSpace()
        self.action_space = defineActionSpace()
        self.evaluating = evaluating
        self.model_filename = model_filename

    def setup(self):
        # Load map data
        with open(self.current_map, 'rb') as f:
            self.space, self.points, self.boundaries, self.screen_size, self.walls = pickle.load(f)
            self.points = self.points[3:len(self.points) - 1]
            self.boundaries = self.boundaries[3:len(self.boundaries) - 1]
            # Set all shapes in space to have collision type 3
            for shape in self.space.shapes:
                shape.collision_type = 3

        # Initialize pygame
        self.screen, self.clock = initialize_pygame(self.screen_size[0], self.screen_size[1])
    
        # Initialize environment variables
        self.observation_size = observation_size
        self.steps_left = self.max_steps
        self.speed_limit = 200
        self.waypoint_reward = 0
        self.collision_penalty = 0
        self.collision_amount = 0
        self.reward_history = []

        # random_positions = self.points[0] + np.random.normal(0, 5, (self.num_agents, 2))
        waypoint_vector = self.points[1] - self.points[0]
        # Add cars to the space
        self.cars = []
        for i in range(self.num_agents):
            car, car_shape = create_car(self.points[0])
            self.cars.append(car)
            self.space.add(car, car_shape)

        # Add the waypoint polys to the space (these should act like sensors that detect when the car passes through them)
        self.waypoint_segments = []
        self.draw_waypoint_segments = []
        for i in range(len(self.points) - 1):
            # Get the points of the polygon from the boundary points
            p1 = self.boundaries[i][0]
            p2 = self.boundaries[i][1]
            p3 = self.boundaries[i + 1][1]
            p4 = self.boundaries[i + 1][0]
            # Create the polygon
            poly = pymunk.Poly(self.space.static_body, ((p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]), (p4[0], p4[1])))
            poly.sensor = True
            poly.collision_type = 2
            self.waypoint_segments.append(poly)
            self.draw_waypoint_segments.append((p1, p2, p3, p4))
            self.space.add(poly)

        # Add collision handler for car and waypoint segments
        self.collision_handler = self.space.add_collision_handler(1, 2)
        self.collision_handler.begin = self.collisionBegin
        self.collision_handler.separate = self.collisionSeparate

        # Add collision handler for car and walls
        self.collision_handler_walls = self.space.add_collision_handler(1, 3)
        self.collision_handler_walls.begin = self.collisionBeginWalls
        self.collision_handler_walls.separate = self.collisionSeparateWalls
        self.collision_handler_walls.post_solve = self.collisionBeginWalls

        # Add collision handler for car and car
        self.collision_handler_car = self.space.add_collision_handler(1, 1)
        self.collision_handler_car.begin = self.collisionBeginCar
        self.collision_handler_car.separate = self.collisionSeparateCar

        angle_to_next_waypoint = np.arctan2(self.points[1][1] - self.points[0][1], self.points[1][0] - self.points[0][0])
        angle_in_degrees = angle_to_next_waypoint * 180 / np.pi

        # Initialize state
        self.state = {
            'positions': [self.points[0] for i in range(self.num_agents)],
            # 'headings': [np.arctan2(self.points[1][1] - self.cars[i].position[1], self.points[1][0] - self.cars[i].position[0]) for i in range(self.num_agents)],
            'headings': [(angle_in_degrees + np.random.normal(0, 20)) * np.pi / 180 for i in range(self.num_agents)],
            'speeds': [0 for i in range(self.num_agents)],
            'current_waypoints': [0 for i in range(self.num_agents)],
            'next_waypoints': [1 for i in range(self.num_agents)],
            'steps_since_last_waypoints': [0 for i in range(self.num_agents)],
            'in_waypoints': [[False for i in range(len(self.points) - 1)] for j in range(self.num_agents)],
            # 'in_waypoints': [[True] + [False for i in range(len(self.points) - 2)] for j in range(self.num_agents)],
            'previous_two_observations': [[np.zeros((3, self.observation_size, self.observation_size)) for i in range(2)] for j in range(self.num_agents)],
            'other_car_collisions': [False for i in range(self.num_agents)],
            'previous_positions': [self.points[0] for i in range(self.num_agents)],
            'distance_to_next_waypoints': [(np.linalg.norm(self.points[1] - self.cars[i].position), i) for i in range(self.num_agents)]
        }

        for i, car in enumerate(self.cars):
            car.velocity = (self.state['speeds'][i]*np.cos(self.state['headings'][i]), self.state['speeds'][i]*np.sin(self.state['headings'][i]))

        # Draw the initial state
        self.screen.fill((255, 0, 0))
        # self.screen.fill((0, 0, 0))
        draw_walls(self.screen, self.walls)
        draw_waypoint_segments(self.screen, self.points, self.state['current_waypoints'][0], self.state['next_waypoints'][0])
        # self.background = self.screen.copy()
        draw_waypoints(self.screen, self.points, self.state['current_waypoints'][0], self.state['next_waypoints'][0])
        # draw_test_waypoints(self.screen, self.draw_waypoint_segments)
        # pygame.display.flip()

    def observation(self, agent_id=0): # Make draw functions dependent on agent_id
        car_pos = self.state['positions'][agent_id]
        # Sort the cars by distance to next waypoint
        self.state['distance_to_next_waypoints'].sort(key=lambda x: x[0])
        # Get the first place car's waypoint
        first_place_waypoint = max(self.state['current_waypoints'])
        cars_in_front = [i for i in range(self.num_agents) if self.state['current_waypoints'][i] == first_place_waypoint]

        # Load the background
        # self.screen.blit(self.background, (0, 0))
        self.screen.fill((255, 0, 0))

        # Erase previous positions of cars
        for i, car in enumerate(self.cars):
            pygame.draw.circle(self.screen, (0, 0, 0), (int(self.state['previous_positions'][i][0]), int(self.state['previous_positions'][i][1])), 5)

        # Draw the waypoint sections
        draw_test_waypoints(self.screen, self.draw_waypoint_segments)

        # Draw other car
        for i, car in enumerate(self.cars):
            if i != agent_id:
                # Draw the car
                pygame.draw.circle(self.screen, (0, 255, 255), (int(car.position[0]), int(car.position[1])), 5)
                arrow_length = max([35 * self.state['speeds'][i] / self.speed_limit, 10])
                arrow_angle = self.state['headings'][i]
                arrow_x = car.position[0] + arrow_length*np.cos(arrow_angle)
                arrow_y = car.position[1] + arrow_length*np.sin(arrow_angle)
                pygame.draw.line(self.screen, (0, 255, 255), car.position, (arrow_x, arrow_y), 3)
                
        # Draw waypoint segments
        draw_waypoint_segments(self.screen, self.points, self.state['current_waypoints'][agent_id], self.state['next_waypoints'][agent_id])
        # Redraw the waypoints
        draw_waypoints(self.screen, self.points, self.state['current_waypoints'][agent_id], self.state['next_waypoints'][agent_id])

        # Draw current car
        # Draw the car
        pygame.draw.circle(self.screen, (255, 255, 0), (int(car_pos[0]), int(car_pos[1])), 5)
        # Draw arrow to show heading and speed
        arrow_length = max([35 * self.state['speeds'][agent_id] / self.speed_limit, 10])
        arrow_angle = self.state['headings'][agent_id]
        arrow_x = car_pos[0] + arrow_length*np.cos(arrow_angle)
        arrow_y = car_pos[1] + arrow_length*np.sin(arrow_angle)
        pygame.draw.line(self.screen, (255, 255, 0), car_pos, (arrow_x, arrow_y), 3)
        # If you are behind the other car, draw a small circle on yourself to indicate that you are behind
        # if current_car_waypoint < other_car_waypoint or (current_car_waypoint == other_car_waypoint and cc_dist_to_next_waypoint > oc_dist_to_next_waypoint):
        if self.state['current_waypoints'][agent_id] < first_place_waypoint or (self.state['current_waypoints'][agent_id] == first_place_waypoint and any([self.state['distance_to_next_waypoints'][agent_id][0] > self.state['distance_to_next_waypoints'][cars_in_front[i]][0] for i in range(len(cars_in_front))])):
            pygame.draw.circle(self.screen, (0, 0, 0), (int(car_pos[0]), int(car_pos[1])), 2)
        if all(self.state['other_car_collisions'][1:]):
            pygame.draw.circle(self.screen, (128, 128, 255), (int(car_pos[0]), int(car_pos[1])), 3)

        # if agent_id == 0:
        #     pygame.display.flip()

        # Get observation (maybe oversample and then downsample for better distance, but worse aliasing)
        oversample_size = observation_size
        # Define sub-surface
        observation = pygame.Surface((oversample_size, oversample_size))
        observation.blit(self.screen, (0, 0), (car_pos[0] - oversample_size / 2, car_pos[1] - oversample_size / 2, oversample_size, oversample_size))
        # observation = pygame.Surface((self.observation_size, self.observation_size))
        # observation.blit(self.screen, (0, 0), (car_pos[0] - self.observation_size / 2, car_pos[1] - self.observation_size / 2, self.observation_size, self.observation_size))
        # Rotate the surface according to the heading
        observation = pygame.transform.rotate(observation, self.state['headings'][agent_id] * 180 / np.pi)
        # Flip the surface vertically
        observation = pygame.transform.flip(observation, True, False)
        # Resize the surface back to the observation size
        observation = pygame.transform.scale(observation, (self.observation_size, self.observation_size))
        observation = pygame.surfarray.pixels3d(observation)

        # Randomly (approximately every 20 frames) do this
        # if np.random.randint(0, 20) == 0:
        #     print("Agent", agent_id)
        #     plt.imshow(observation)
        #     plt.show()

        # Convert to CxHxW
        observation = np.transpose(observation, (2, 0, 1))

        return observation

    def reset(self):
        maps = self.maps
        max_steps = self.max_steps
        num_agents = self.num_agents
        current_map = self.current_map
        other_agent = self.other_agent
        evaluating = self.evaluating
        model_filename = self.model_filename
        # Clear all self variables (except maps)
        self.__dict__.clear()
        self.maps = maps
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.current_map = current_map
        if evaluating:
            self.other_agent = PPO.load('../eval_models/' + model_filename)
            # self.other_agent = PPO.load('../eval_models/good_models/comp_model.zip')
        else:
            # self.other_agent = other_agent
            self.other_agent = PPO.load('../eval_models/temp_model.zip')
        self.observation_space = defineObservationSpace()
        self.action_space = defineActionSpace()
        self.evaluating = evaluating
        self.model_filename = model_filename


        random_map = np.random.choice(self.maps)
        # Update map to next map
        if self.current_map is None:
            self.current_map = self.maps[0]
        elif self.evaluating:
            self.current_map = self.maps[(self.maps.index(self.current_map) + 1) % len(self.maps)]
            # self.other_agent = PPO.load('../eval_models/temp_model.zip')
        else:
            self.current_map = random_map
        self.setup()

        return self.observation(0)

    def step(self, action):
        # Establish reward (penalty for living)
        reward = -0.01

        model = self.other_agent
        self.state['distance_to_next_waypoints'] = [(np.linalg.norm(self.points[self.state['next_waypoints'][i]] - self.cars[i].position), i) for i in range(self.num_agents)]

        # For each car,
        for i, car in enumerate(self.cars):
            steer, throttle = action
            # print("Player action:", action)
            if i != 0 and not(self.state['other_car_collisions'][i]):
                # Get action from model
                current_obs = self.observation(i)

                # Create a 3-stack of observations
                three_stack_obs = np.concatenate((self.state['previous_two_observations'][i][0], self.state['previous_two_observations'][i][1], current_obs), axis=0)
                # three_stack_obs = current_obs

                other_action, _states = model.predict(three_stack_obs.copy(), deterministic=True)
                steer, throttle = other_action
                # print("Action:", other_action)

                # Update previous two observations
                self.state['previous_two_observations'][i][0] = self.state['previous_two_observations'][i][1]
                self.state['previous_two_observations'][i][1] = current_obs
            # else:
            #     if i != 0:
            #         print("Not predicting for car", i, "because of collision")
            if not(self.state['other_car_collisions'][i]):
                # Get new heading
                new_heading = getNewHeading(self.state['headings'][i], steer)
                self.state['headings'][i] = new_heading
                # Get new speed
                new_speed = getNewSpeed(self.state['speeds'][i], throttle, self.speed_limit)
                self.state['speeds'][i] = new_speed
                # Update car velocity
                car.velocity = (self.state['speeds'][i]*np.cos(self.state['headings'][i]), self.state['speeds'][i]*np.sin(self.state['headings'][i]))

                # If close to the other car, gain a velocity bonus because of drafting
                # if abs(distance_to_other_car) < 20:
                #     self.speed_limit = 200
                #     reward = 0
                # else:
                #     if self.speed_limit > 30:
                #         self.speed_limit = max([30, self.speed_limit * 0.99])
                    # car.velocity = (car.velocity[0] * 2, car.velocity[1] * 2)

        # if self.speed_limit > 30 and abs(distance_to_other_car) > 20:
        #     print("Speed limit:", self.speed_limit)

        # Update space
        self.space.step(1/FPS)

        # Update state
        for i, car in enumerate(self.cars):
            self.state['previous_positions'][i] = self.state['positions'][i]
            self.state['positions'][i] = car.position

        # Update steps left
        self.steps_left -= 1
        # self.state['steps_since_last_waypoints'][0] += 1

        # If in an earlier waypoint than the other car or behind it, increment time step since last waypoint
        # current_car_waypoint = self.state['current_waypoints'][0]
        # other_car_waypoint = self.state['current_waypoints'][1]
        # cc_dist_to_next_waypoint = np.linalg.norm(self.state['positions'][0] - self.points[self.state['next_waypoints'][0]])
        # oc_dist_to_next_waypoint = np.linalg.norm(self.state['positions'][1] - self.points[self.state['next_waypoints'][1]])

        # if current_car_waypoint < other_car_waypoint or (current_car_waypoint == other_car_waypoint and cc_dist_to_next_waypoint > oc_dist_to_next_waypoint):
        #     self.state['steps_since_last_waypoints'][0] += 1

        ### Reward Shaping ###
        # Check for reward gained from passing through waypoints
        reward += self.waypoint_reward
        # if self.waypoint_reward > 0:
        #     # Penalize the agent for moving too far away from other agent (scale to)
        #     # If the distance is greater than 30, penalize the agent
        #     if distance_to_other_car < 30 and not(self.state['other_car_collision']):
        #         reward = reward * 2
        self.waypoint_reward = 0

        # Check for wall collision penalty
        reward += self.collision_penalty
        # if self.collision_penalty < 0:
        #     self.collision_amount += 1
            # print("hey")
        self.collision_penalty = 0

        # Check for done
        checks = [
            # Run out of steps
            self.steps_left <= 0,
            # Reached the end of the waypoints
            self.state['current_waypoints'][0] >= len(self.points) - 3,
            # Haven't passed through a waypoint in a while
            self.state['steps_since_last_waypoints'][0] > 100,
            # Collision with wall
            self.collision_penalty < 0
        ]
        done = any(checks)

        # if (done):
        #     print(checks)

        # if checks[2] or checks[0]:
        if checks[0] or checks[2] or checks[3]:
            # If run out of steps or haven't passed through a waypoint in a while, penalize the agent by removing the reward gained from passing through waypoints
            # total_reward = sum(self.reward_history)
            # # print("Total reward:", total_reward)
            # if total_reward > 0:
            #     reward -= total_reward * 0.5
            reward -= 50
        elif checks[1]:
            reward += 100

        observation = self.observation(0)
        # pygame.display.flip()
        # self.clock.tick(30)

        # if abs(reward) > 0.01:
        #     print("Reward:", reward)

        # Update reward history
        if reward > 0:
            self.reward_history.append(reward)

        # Return observation, reward, done, info
        return observation, reward, done, {}

    def render(self):
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset()
                
            # Update the space
            self.space.step(1/FPS)

            # Clear the pygame screen
            self.screen.fill((0, 0, 0))

            # Draw waypoints
            draw_waypoints(self.screen, self.points, self.state['current_waypoint'], self.state['next_waypoint'])
            # Draw waypoint segments
            draw_waypoint_segments(self.screen, self.points)
            # Draw walls
            draw_walls(self.screen, self.walls)


            # Draw car
            car_pos = self.car.position
            pygame.draw.circle(self.screen, (255, 255, 0), (int(car_pos[0]), int(car_pos[1])), 2)

            # Draw arrow to show heading and speed
            arrow_length = 10
            arrow_angle = self.state['heading']
            arrow_x = car_pos[0] + arrow_length*np.cos(arrow_angle)
            arrow_y = car_pos[1] + arrow_length*np.sin(arrow_angle)
            pygame.draw.line(self.screen, (255, 255, 0), car_pos, (arrow_x, arrow_y))

            # Update screen
            pygame.display.flip()
            self.clock.tick(FPS)

    def close(self):
        pass

    def collisionBegin(self, arbiter, space, data):
        # If this is the first time the car has collided with this waypoint segment,
        # if arbiter.is_first_contact:
        if self.steps_left == self.max_steps:
            return False
        # Use the waypoint segment's index to determine which waypoint the car has passed through
        waypoint_index = self.waypoint_segments.index(arbiter.shapes[1])-1
        # print("Collision with waypoint segment", waypoint_index)
        car_index = self.cars.index(arbiter.shapes[0].body)

        # print("Collision with waypoint segment", waypoint_index)
        self.state['in_waypoints'][car_index][waypoint_index] = True

        if waypoint_index > self.state['current_waypoints'][car_index]:
            if car_index == 0:
                # Sort the cars by distance to next waypoint
                self.state['distance_to_next_waypoints'].sort(key=lambda x: x[0])
                # Get the first place car's waypoint
                first_place_waypoint = max(self.state['current_waypoints'])
                cars_in_front = [i for i in range(self.num_agents) if self.state['current_waypoints'][i] == first_place_waypoint]

                # If you are in first place, reward yourself
                if not(self.state['current_waypoints'][0] < first_place_waypoint or (self.state['current_waypoints'][0] == first_place_waypoint and any([self.state['distance_to_next_waypoints'][0][0] > self.state['distance_to_next_waypoints'][cars_in_front[i]][0] for i in range(len(cars_in_front))]))):
                    self.waypoint_reward = 5 * (waypoint_index - self.state['current_waypoints'][car_index])
                else:
                    self.waypoint_reward = 2.5 * (waypoint_index - self.state['current_waypoints'][car_index])
                # living_agents = [i for i in range(self.num_agents) if not(self.state['other_car_collisions'][i])]
                # living_agents_in_same_waypoint = [i for i in living_agents if self.state['current_waypoints'][i] == self.state['current_waypoints'][0]]
                # # Multiply the reward by the fraction of living agents that are in the same waypoint (to encourage cooperation)
                # # self.waypoint_reward *= len(living_agents_in_same_waypoint) / len(living_agents)
            self.state['current_waypoints'][car_index] = waypoint_index
            self.state['next_waypoints'][car_index] = waypoint_index + 1
            self.state['steps_since_last_waypoints'][car_index] = 0
            # print(arbiter.shapes[0], arbiter.shapes[1])
        return False
    
    def collisionSeparate(self, arbiter, space, data):
        # print("Separation with waypoint segment", waypoint_index)
        if not(hasattr(self, 'waypoint_segments')):
            return True
        elif arbiter.shapes[1] not in self.waypoint_segments:
            return True
        car_index = self.cars.index(arbiter.shapes[0].body)
        waypoint_index = self.waypoint_segments.index(arbiter.shapes[1])-1
        self.state['in_waypoints'][car_index][waypoint_index] = False
        return True

    def collisionBeginWalls(self, arbiter, space, data):
        car_index = self.cars.index(arbiter.shapes[0].body)
        # If first time step or in waypoint, ignore collision
        if self.steps_left == self.max_steps or any(self.state['in_waypoints'][car_index]):
            self.state['speeds'][car_index] = 0
            # if car_index == 0:
            #     self.collision_penalty = -1
            return True

        if car_index == 0:
            num_waypoints_passed = max([1,self.state['current_waypoints'][car_index]])
            # self.collision_penalty = -2.5 * num_waypoints_passed # FIX
            self.collision_penalty = -1
            # self.collision_penalty = -0.1
        else:
            self.state['other_car_collisions'][car_index] = True
            # Reset the other car to the current waypoint
            self.cars[car_index].velocity = (0, 0)
            self.state['speeds'][car_index] = 0
        return True

    def collisionSeparateWalls(self, arbiter, space, data):
        # print("Separation with wall")
        self.collision_penalty = 0 # FIX
        return True
    
    def collisionBeginCar(self, arbiter, space, data):
        # Don't process collisions between cars
        # Half the speed of both cars
        car_1 = arbiter.shapes[0].body
        car_2 = arbiter.shapes[1].body
        car_index_1 = self.cars.index(car_1)
        car_index_2 = self.cars.index(car_2)
        self.state['speeds'][car_index_1] = 0
        self.state['speeds'][car_index_2] = 0
        # print("Collision with car")
        return True

    def collisionSeparateCar(self, arbiter, space, data):
        return True

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

### Display Episodes in PyGame ###
def playNEpisodes(n, env, model, max_steps=1000):
    for episode in range(n):
        if episode == 0:
            obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action, _states = model.predict(obs.copy(), deterministic=True)
            # print("Action:", action)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            pygame.display.update()
            # pygame.time.Clock().tick(120)

            if done:
                print(f'Episode {episode} finished after {step} steps with reward {total_reward}')
                # Pause the game
                pause = True
                while pause:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                pause = False
                                break
                break

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
class CustomCallback(BaseCallback):
    """
    Custom Callback that should be called after a new best model is saved.
    I want this callback to update the environment's other_agent variable with the new model so the agent is training against the most recent model.
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.other_agent = None

    def _on_step(self) -> bool:
        if self.other_agent is None:
            self.other_agent = self.model
        else:
            self.other_agent = self.model

        # Update the environment's other_agent variable
        # self.training_env.set_attr('other_agent', self.other_agent)

        return True
    
class EveryUpdateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EveryUpdateCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        # Get access to the parent callback info
        parent_callback = self.parent

        # Save the model
        self.model.save('../eval_models/temp_model')
        self.training_env.set_attr('other_agent', self.model)
        return True


### Main ###
if __name__ == "__main__":
    # maps = ["../maps/map_10_30_800_800.pkl"]#, "../maps/map_50_70_800_800.pkl", "../maps/map_70_90_800_800.pkl"]
    # maps = ["../maps/map_30_50_800_800.pkl"]
    maps = ["../maps/map_70_90_800_800.pkl"]

    # Define observation and action spaces
    old_env = RacingEnv(maps, max_steps, num_agents)
    old_env = make_vec_env(lambda: old_env, n_envs=1, seed=np.random.randint(0, 10000))
    old_env = VecFrameStack(old_env, n_stack=3)
    observation_space = old_env.observation_space
    action_space = old_env.action_space

    # Load the pretrained model
    # pretrained_model = PPO.load('../eval_models/best_model_temp.zip', env=old_env, custom_objects={'observation_space': observation_space, 'action_space': action_space}, device="cuda")
    pretrained_model = PPO("CnnPolicy", old_env, verbose=1, device="cuda")

    # Save the pretrained model to temp_model.zip
    pretrained_model.save('../eval_models/temp_model')

    # Initialize environment
    env = RacingEnv(maps, max_steps, num_agents, pretrained_model)
    # Parallelize environment
    vec_env = make_vec_env(lambda: env, n_envs=n_envs, seed=np.random.randint(0, 10000))
    # Frame stack
    vec_env = VecFrameStack(vec_env, n_stack=3)

    # Policy Args
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=32)
    # )

    # Create model
    model = PPO("CnnPolicy", vec_env, verbose=1, device="cuda", **hyperparameters, tensorboard_log="../logs/")
    # model = PPO.load('../eval_models/best_model.zip', env=vec_env, device="cuda", **hyperparameters, tensorboard_log="../logs/")
    # model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    # Callback envs
    eval_env = RacingEnv(maps, max_steps, num_agents, pretrained_model)
    eval_env = make_vec_env(lambda: eval_env, n_envs=n_envs, seed=np.random.randint(0, 10000))
    eval_env = VecFrameStack(eval_env, n_stack=3)

    # Callbacks
    eval_callback = EvalCallback(eval_env, best_model_save_path='../eval_models/', log_path='../logs/', eval_freq=10000, deterministic=True, render=False, verbose=1, callback_on_new_best=CustomCallback(), callback_after_eval=EveryUpdateCallback())

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True, tb_log_name="run_1")

    print("Hello, world!")