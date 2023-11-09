### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from typing import Callable
import time

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
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback

# Simulation (pygame, pymunk)
import pygame
import pygame.gfxdraw
import pymunk
import pickle

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData
from CustomRacing2D_env import draw_waypoints

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

def draw_waypoint_segments(screen, points):
    for i in range(len(points) - 1):
        pygame.draw.line(screen, (255, 255, 255), points[i], points[i + 1], 2)

def draw_test_waypoints(screen, points):
    for p1, p2, p3, p4 in points:
        random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        pygame.draw.polygon(screen, random_color, (p1, p2, p3, p4))

def getNewHeading(heading_angle, steering_angle):
    '''
    Returns the new heading of the agent based on the current heading angle and steering angle.
    In other words, just adds the steering angle to the heading angle.
    '''
    # Add some noise to the steering angle
    noise = np.random.normal(0, 0.05)
    steering_angle += noise
    # Resize steering angle between -pi/128 and pi/128
    steering_angle = steering_angle * np.pi / 128
    new_angle = heading_angle - steering_angle
    # Keep the angle between -pi and pi
    if (new_angle > np.pi):
        new_angle -= 2 * np.pi
    elif (new_angle < -np.pi):
        new_angle += 2 * np.pi
    return new_angle

def getNewSpeed(speed, throttle, speed_limit):
    force = throttle

    force += np.random.normal(0, 0.05)

    new_speed = speed + force

    if new_speed > speed_limit:
        new_speed = speed_limit
    elif new_speed < 30:
        new_speed = 30
    
    return new_speed

### Global Variables ###
# Model parameters
max_steps = 5000
total_timesteps = 1000000
observation_size = 64

# Pygame parameters
FPS = 120

# Simulation parameters


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
    def __init__(self, maps, max_steps):
        # Save maps
        self.maps = maps
        self.max_steps = max_steps

        self.reset()

    def setup(self, map, max_steps):
        # Load map data
        with open(map, 'rb') as f:
            self.space, self.points, self.boundaries, self.screen_size, self.walls = pickle.load(f)
            self.points = self.points[1:len(self.points) - 1]
            self.boundaries = self.boundaries[1:len(self.boundaries) - 1]
            # Set all shapes in space to have collision type 3
            for shape in self.space.shapes:
                shape.collision_type = 3
                # If one of the walls has a weirdly large area, remove it
                if shape.area > 10000:
                    self.space.remove(shape)

        # Initialize pygame
        self.screen, self.clock = initialize_pygame(self.screen_size[0], self.screen_size[1])
    
        # Initialize environment variables
        self.observation_space = defineObservationSpace()
        self.action_space = defineActionSpace()
        self.observation_size = observation_size
        self.max_steps = max_steps
        self.waypoint_reward = 0
        self.collision_penalty = 0
        # self.reward_range = (-np.inf, np.inf)

        # Add the car to the space
        self.car, self.car_shape = create_car(self.points[0])
        self.space.add(self.car, self.car_shape)

        # Add the waypoint polys to the space (these should act like sensors that detect when the car passes through them)
        self.waypoint_segments = []
        self.draw_waypoint_segments = []
        for i in range(len(self.points) - 1):
            scale = 50
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

        # Initialize state
        self.state = {
            'position': self.points[0],
            'heading': np.arctan2(self.points[1][1] - self.points[0][1], self.points[1][0] - self.points[0][0]),
            'speed': 0,
            'current_waypoint': 0,
            'next_waypoint': 1,
            'steps_since_last_waypoint': 0
        }

        self.car.velocity = (self.state['speed']*np.cos(self.state['heading']), self.state['speed']*np.sin(self.state['heading']))

        # Draw the initial state
        self.screen.fill((0, 0, 0))
        draw_waypoints(self.screen, self.points, self.state['current_waypoint'], self.state['next_waypoint'])
        draw_walls(self.screen, self.walls)
        draw_waypoint_segments(self.screen, self.points)
        # draw_test_waypoints(self.screen, self.draw_waypoint_segments)
        pygame.display.flip()

    def observation(self):
        # Get car position
        car_pos = self.car.position

        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Redraw the walls
        draw_walls(self.screen, self.walls)
        # Draw waypoint segments
        draw_waypoint_segments(self.screen, self.points)
        # Redraw the waypoints
        draw_waypoints(self.screen, self.points, self.state['current_waypoint'], self.state['next_waypoint'])

        # Draw arrow to show heading and speed
        # arrow_length = 20
        # arrow_angle = self.state['heading']
        # arrow_x = car_pos[0] + arrow_length*np.cos(arrow_angle)
        # arrow_y = car_pos[1] + arrow_length*np.sin(arrow_angle)
        # pygame.draw.line(self.screen, (255, 255, 0), car_pos, (arrow_x, arrow_y), width=4)
        # Draw the new car position
        pygame.draw.circle(self.screen, (255, 255, 0), (int(car_pos[0]), int(car_pos[1])), 5)


        # Get observation (100x100 image around the car)
        # Define sub-surface
        observation = pygame.Surface((self.observation_size, self.observation_size))
        observation.blit(self.screen, (0, 0), (car_pos[0] - self.observation_size / 2, car_pos[1] - self.observation_size / 2, self.observation_size, self.observation_size))
        # Rotate the surface according to the heading
        observation = pygame.transform.rotate(observation, self.state['heading'] * 180 / np.pi)
        # Resize the surface back to 100x100
        observation = pygame.transform.scale(observation, (self.observation_size, self.observation_size))
        # Flip the surface vertically
        observation = pygame.transform.flip(observation, True, False)
        observation = pygame.surfarray.pixels3d(observation)

        # plt.imshow(observation)
        # plt.show()
        # exit()

        # Convert to CxHxW
        observation = np.transpose(observation, (2, 0, 1))


        return observation

    def reset(self):
        maps = self.maps
        max_steps = self.max_steps
        # Clear all self variables (except maps)
        self.__dict__.clear()
        self.maps = maps
        self.max_steps = max_steps

        # Pick a new map
        map = np.random.choice(self.maps)
        self.setup(map, self.max_steps)
        self.car.velocity = (self.state['speed']*np.cos(self.state['heading']), self.state['speed']*np.sin(self.state['heading']))

        self.steps_left = self.max_steps
        self.speed_limit = 200

        return self.observation()

    def step(self, action):
        # Establish reward (penalty for living)
        reward = -0.01
        self.state['steps_since_last_waypoint'] += 1

        # Take action
        steer, throttle = action

        # Get new heading
        new_heading = getNewHeading(self.state['heading'], steer)
        self.state['heading'] = new_heading

        # Get new speed
        new_speed = getNewSpeed(self.state['speed'], throttle, self.speed_limit)
        self.state['speed'] = new_speed

        # Update car velocity
        self.car.velocity = (self.state['speed']*np.cos(self.state['heading']), self.state['speed']*np.sin(self.state['heading']))

        # Update space
        self.space.step(1/FPS)
        # pygame.display.flip()
        # self.clock.tick(FPS)

        # Update state
        self.state['position'] = self.car.position

        # Update steps left
        self.steps_left -= 1

        # Check for reward gained from passing through waypoints
        reward += self.waypoint_reward
        self.waypoint_reward = 0

        # Check for wall collision penalty
        reward += self.collision_penalty

        # Check for done
        checks = [
            # Run out of steps
            self.steps_left <= 0,
            # Reached the end of the waypoints
            self.state['current_waypoint'] == len(self.points) - 2,
            # Haven't passed through a waypoint in a while
            self.state['steps_since_last_waypoint'] > 500,
            # Collision with wall
            self.collision_penalty < 0
        ]
        done = any(checks)

        if (done):
            print(checks)

        if checks[2] or checks[0]:
            reward -= max([10, 4 * self.state['current_waypoint']])
        elif checks[1]:
            reward += 100

        observation = self.observation()

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
        # Use the waypoint segment's index to determine which waypoint the car has passed through
        waypoint_index = self.waypoint_segments.index(arbiter.shapes[1])

        # print("Collision with waypoint segment", waypoint_index)

        if waypoint_index > self.state['current_waypoint']:
            self.waypoint_reward = 5 * (waypoint_index - self.state['current_waypoint'])
            self.state['current_waypoint'] = waypoint_index
            self.state['next_waypoint'] = waypoint_index + 1
            self.state['steps_since_last_waypoint'] = 0
            # print(arbiter.shapes[0], arbiter.shapes[1])
        return True
    
    def collisionSeparate(self, arbiter, space, data):
        # print("Separation with waypoint segment", waypoint_index)
        return True

    def collisionBeginWalls(self, arbiter, space, data):
        # If first time step, ignore collision
        if self.steps_left == self.max_steps:
            return False

        num_waypoints_passed = self.state['current_waypoint']
        self.collision_penalty = min([-10, -4 * num_waypoints_passed])
        return True

    def collisionSeparateWalls(self, arbiter, space, data):
        self.collision_penalty = 0
        return True

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

### Environment Functions ###
def create_car(pos):
    car = pymunk.Body(10, 1, body_type=pymunk.Body.DYNAMIC)
    car.position = (pos[0], pos[1])
    car_shape = pymunk.Circle(car, 2)
    car_shape.elasticity = 1
    car_shape.collision_type = 1
    return car, car_shape

### Display Episodes in PyGame ###
def playNEpisodes(n, env, model, max_steps=1000):
    for episode in range(n):
        obs = env.reset()
        for step in range(max_steps):
            action, _states = model.predict(obs.copy(), deterministic=True)
            # print("Action:", action)
            obs, reward, done, info = env.step(action)

            pygame.display.update()
            if done:
                print(f'Episode {episode} finished after {step} steps')
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
    

### Main ###
if __name__ == "__main__":
    maps = ["../maps/map_10_30_800_800.pkl", "../maps/map_30_50_800_800.pkl"]
    # Initialize environment
    env = RacingEnv(maps, max_steps)

    # Parallelize environment
    vec_env = make_vec_env(lambda: env, n_envs=1, seed=np.random.randint(0, 10000))

    # Frame stack
    vec_env = VecFrameStack(vec_env, n_stack=3)

    # Policy Args
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32)
    )

    # Create model
    # model = PPO("CnnPolicy", vec_env, verbose=1, device="cpu")
    model = PPO("CnnPolicy", vec_env, verbose=1, device="cuda")
    # model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    # Callback env
    eval_env = RacingEnv(maps, max_steps)
    eval_env = make_vec_env(lambda: eval_env, n_envs=1, seed=np.random.randint(0, 10000))
    eval_env = VecFrameStack(eval_env, n_stack=3)

    # Callbacks
    eval_callback = EvalCallback(eval_env, best_model_save_path='../eval_models/', log_path='../logs/', eval_freq=5000, deterministic=True, render=False, verbose=1)

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # Save model
    model.save('../models/temp_model')

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

    # Render n episodes
    vec_env.reset()
    playNEpisodes(5, vec_env, model, max_steps)


    print("Hello, world!")