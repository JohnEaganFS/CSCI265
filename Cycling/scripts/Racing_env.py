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

def draw_waypoint_segments(screen, points):
    for i in range(len(points) - 1):
        pygame.draw.line(screen, (0, 0, 255), points[i], points[i + 1])

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
observation_size = 100

# Pygame parameters
FPS = 120

# Simulation parameters


### Observation Space ###
# Observation space is a 2D array of pixel values (image around the car)
def defineObservationSpace():
    return gym.spaces.Box(low=0, high=255, shape=(observation_size, observation_size, 3), dtype=np.uint8)

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
            self.space, self.points, self.boundaries, self.screen_size, self.walls = pickle.load(f)
            self.points = self.points[1:]

        # Initialize pygame
        self.screen, self.clock = initialize_pygame(self.screen_size[0], self.screen_size[1])
    
        # Initialize environment variables
        self.observation_space = defineObservationSpace()
        self.action_space = defineActionSpace()
        self.observation_size = observation_size
        self.max_steps = max_steps
        self.waypoint_reward = 0
        # self.reward_range = (-np.inf, np.inf)

        # Add the car to the space
        self.car, self.car_shape = create_car(self.points[0])
        self.space.add(self.car, self.car_shape)

        # Add the waypoint polys to the space (these should act like sensors that detect when the car passes through them)
        self.waypoint_segments = []
        for i in range(len(self.points) - 1):
            scale = 50
            # Get the vector from the current waypoint to the next waypoint
            vec = np.array(self.points[i + 1]) - np.array(self.points[i])
            # Get the unit vector
            unit_vec = vec/np.linalg.norm(vec)
            # Get the perpendicular vector
            perp_vec = np.array([-unit_vec[1], unit_vec[0]])
            # Get the points of the polygon
            p1 = self.points[i] + scale*perp_vec
            p2 = self.points[i] - scale*perp_vec
            p3 = self.points[i + 1] - scale*perp_vec
            p4 = self.points[i + 1] + scale*perp_vec
            # Create the polygon
            poly = pymunk.Poly(self.space.static_body, ((p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]), (p4[0], p4[1])))
            poly.sensor = True
            poly.collision_type = 2
            self.waypoint_segments.append(poly)
            self.space.add(poly)


        # Add collision handler
        self.collision_handler = self.space.add_collision_handler(1, 2)
        self.collision_handler.begin = self.collisionBegin
        self.collision_handler.separate = self.collisionSeparate

        self.state = {
            'position': self.points[0],
            'heading': np.arctan2(self.points[1][1] - self.points[0][1], self.points[1][0] - self.points[0][0]),
            'speed': 50,
            'current_waypoint': 0,
            'next_waypoint': 1
        }

        self.initial_state = self.state.copy()
        self.car.velocity = (self.state['speed']*np.cos(self.state['heading']), self.state['speed']*np.sin(self.state['heading']))

        # self.observation()

        self.render()



    def observation(self):
        # Clear the pygame screen
        self.screen.fill((0, 0, 0))

        # Draw waypoints
        draw_waypoints(self.screen, self.points, self.state['current_waypoint'], self.state['next_waypoint'])
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

        # Get observation (100x100 image around the car)
        # Define sub-surface
        observation = pygame.Surface((self.observation_size, self.observation_size))
        observation.blit(self.screen, (0, 0), (car_pos[0] - self.observation_size/2, car_pos[1] - self.observation_size/2, self.observation_size, self.observation_size))
        # Convert to numpy array
        observation = pygame.surfarray.array3d(observation)
        # Convert to RGB
        observation = np.flip(observation, axis=0)
        observation = np.rot90(observation, k=3)
        # Convert to uint8
        observation = observation.astype(np.uint8)
        
        # Display observation
        # plt.imshow(observation)
        # plt.show()

        return observation

    def reset(self):
        self.car.position = self.points[0][0], self.points[0][1]
        self.waypoint_reward = 0
    
        self.state = self.initial_state.copy()
        self.car.velocity = (self.state['speed']*np.cos(self.state['heading']), self.state['speed']*np.sin(self.state['heading']))

        self.steps_left = self.max_steps
        self.speed_limit = 200

        return self.observation()

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

        print("Collision with waypoint segment", waypoint_index)

        if waypoint_index > self.state['current_waypoint']:
            self.waypoint_reward = 5 * (waypoint_index - self.state['current_waypoint'])
            self.state['current_waypoint'] = waypoint_index
            self.state['next_waypoint'] = waypoint_index + 1
            # print(arbiter.shapes[0], arbiter.shapes[1])
        return True
    
    def collisionSeparate(self, arbiter, space, data):
        waypoint_index = self.waypoint_segments.index(arbiter.shapes[1])
        # print("Separation with waypoint segment", waypoint_index)
        return True

### Environment Functions ###
def create_car(pos):
    car = pymunk.Body(10, 1, body_type=pymunk.Body.DYNAMIC)
    car.position = (pos[0], pos[1])
    car_shape = pymunk.Circle(car, 2)
    car_shape.elasticity = 1
    car_shape.collision_type = 1
    return car, car_shape


### Main ###
if __name__ == "__main__":

    # Initialize environment
    env = RacingEnv("../maps/map_10_30_800_800.pkl")



    print("Hello, world!")