'''
Switching to a 2D physics engine library for the sake of simplicity. I will be using PyMunk. This will enable be me to
create the "walls" of the track and the "car" that will be racing around the track.
'''

### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from typing import Callable, Dict, List, Optional, Tuple, Union

# Gym
import gym

# SB3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

# Simulation (pygame, pymunk)
import pygame
import pymunk

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData

### Miscellaneous Functions ###
def boundaries(x, y, scale):
    '''
    Returns the boundaries of the environment as a list of 4 points
    '''
    v = y - x
    
    p1 = np.array([v[1], -v[0]])
    p2 = np.array([-v[1], v[0]])

    # Scale to length of v
    p1 = p1 / np.linalg.norm(p1) * scale * 0.5
    p2 = p2 / np.linalg.norm(p2) * scale * 0.5

    p3 = p1 + v
    p4 = p2 + v

    points = np.array([p1, p2, p3, p4])
    points += x
    return points
    
def define_boundaries(points, scale):
    # Initialize boundary_points
    boundary_points = []

    # Create boundaries
    for i in range(len(points) - 1):
        A = np.array([points[i][0], points[i][1]])
        B = np.array([points[i + 1][0], points[i + 1][1]])
        a, b, c, d = boundaries(A, B, scale)
        # For the first point, define itself and the next point
        if i == 0:
            boundary_points.append((a, b))
        # For all other points, only define the next point (the current point has already been defined by the previous iteration)
        boundary_points.append((c, d))
    
    return boundary_points

def inRectangle(point, boundary_points):
    ''' 
    Check if a point is in a rectangle defined by the boundary points.
    '''
    # Get the boundary points
    a, b, d, c = boundary_points[0], boundary_points[1], boundary_points[2], boundary_points[3]
    # Check if point is in rectangle
    vector_ap = point - a
    vector_ab = b - a
    vector_ad = d - a
    return 0 <= np.dot(vector_ap, vector_ab) <= np.dot(vector_ab, vector_ab) and 0 <= np.dot(vector_ap, vector_ad) <= np.dot(vector_ad, vector_ad)

def inPoly(point, path_points):
    '''
    Check if a point is in a polygon defined by the boundary points.
    '''
    # Get the boundary points
    a, b, c, d = path_points[0], path_points[1], path_points[2], path_points[3]
    # Check if point is in polygon
    path = mplPath.Path(np.array([a, b, d, c]))
    return path.contains_point(point)

### Global Variables ###
max_steps = 2000
total_timesteps = 300000
n = 1

 # Create pymunk space
space = pymunk.Space()

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
FPS = 120

# Pull waypoints from .gpx file
points = removeDuplicatePoints(read_gpx('../gpx/windy_road.gpx', 1))
points = points[70:90]
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
# print(boundary_points)


### Observations ###
observations = ['vector_x', 'vector_y', 'speed', 'heading', 'sensor_front', 'sensor_left', 'sensor_right', 'past_sensor_front', 'past_sensor_left', 'past_sensor_right']

def defineObservationSpace():
    lows = {
        'vector_x': -np.inf,
        'vector_y': -np.inf,
        'speed': 0,
        'heading': -np.pi,
        'sensor_front': -1,
        'sensor_left': -1,
        'sensor_right': -1,
        'past_sensor_front': -1,
        'past_sensor_left': -1,
        'past_sensor_right': -1,
    }
    highs = {
        'vector_x': np.inf,
        'vector_y': np.inf,
        'speed': 10,
        'heading': np.pi,
        'sensor_front': np.inf,
        'sensor_left': np.inf,
        'sensor_right': np.inf,
        'past_sensor_front': np.inf,
        'past_sensor_left': np.inf,
        'past_sensor_right': np.inf,
    }
    low = np.array([lows[obs] for obs in observations])
    high = np.array([highs[obs] for obs in observations])
    return gym.spaces.Box(low, high, dtype=np.float32)

# observations = ['vec', 'speed', 'heading', 'sensors']

# def defineObservationSpace():
#     return gym.spaces.Dict(
#         spaces = {
#             "vec": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
#             "speed": gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
#             "heading": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
#             "sensors": gym.spaces.Box(low=-1, high=np.inf, shape=(3,), dtype=np.float32),
#         }
#     )

### Actions ###
actions = ['steer', 'throttle']

def defineActionSpace():
    lows = {
        'steer': -1,
        'throttle': -1
        # 'brake': 0
    }
    highs = {
        'steer': 1,
        'throttle': 1
        # 'brake': 1
    }
    low = np.array([lows[act] for act in actions])
    high = np.array([highs[act] for act in actions])
    return gym.spaces.Box(low, high, dtype=np.float32)

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

def getNewSpeed(speed, throttle, speed_limit, vel, old_vel):
    '''
    Throttle controls acceleration
    Brake controls deceleration
    '''
    # Get force from throttle and brake
    force = throttle #  - brake

    # Add some noise to the force +- 0.05
    force += np.random.normal(0, 0.05)

    # If braking, increase force
    # if (force < 0):
    #     force *= 5

    # Update speed
    new_speed = speed + force

    # Keep speed between -speed_limit and speed_limit
    if (new_speed > speed_limit):
        new_speed = speed_limit
    elif (new_speed < 0):
        new_speed = 0

    # If velocity has changed from old velocity, a collision has occurred (set speed to 0)
    did_collide = False
    if (abs(vel - old_vel) > 10):
        did_collide = True

    return new_speed, did_collide

def sensorReading(sensor, car, space, sensor_range, heading):
    # Get sensor readings
    s_range = sensor_range

    sensor_vector = car.position + pymunk.Vec2d(s_range * np.cos(heading + sensor), s_range * np.sin(heading + sensor))
    sensor_hit = space.segment_query(car.position, sensor_vector, 0, pymunk.ShapeFilter(categories=0b1, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1))
    if sensor_hit:
        sensor_vector = sensor_hit[0].point
        sensor_reading = np.linalg.norm(sensor_vector - car.position)
    else:
        sensor_reading = 1000
    return sensor_reading

def inPreviousWaypoints(boundaries, current_waypoint, car_position):
    previousWaypoints = boundaries[:current_waypoint]
    for i in range(len(previousWaypoints) - 1):
        a, b, c, d = previousWaypoints[i][0], previousWaypoints[i][1], previousWaypoints[i + 1][0], previousWaypoints[i + 1][1]
        if inPoly(car_position, [a, b, c, d]):
            return True
    return False 

### Environment ###
class CustomRacing2DEnv(gym.Env):
    def __init__(self, points=points, boundaries=boundary_points):
        self.observations = observations
        self.observation_space = defineObservationSpace()
        self.actions = actions
        self.action_space = defineActionSpace()
        self.max_steps = max_steps
        self.original_points = points
        self.boundaries = boundaries
        self.num_episodes = 0
        self.has_reset_screen = False

    def reset(self):
        # Increment number of episodes
        self.num_episodes += 1
        self.has_reset_screen = False
        self.reset_count = 0

        waypoints = self.original_points.copy()
        # full_boundary_points = np.array([boundaries(waypoints[i], waypoints[i + 1], min_vector_length) for i in range(len(waypoints) - 1)])

        # Create new game
        space = pymunk.Space()
        space, cars, walls, walls_draw = createGame(space, self.boundaries, self.original_points)

        v = waypoints[1] - waypoints[0]
        self.state = {
            'vector_x': v[0],
            'vector_y': v[1],
            'speed': 0,
            # Start with random heading
            'heading': np.random.uniform(-np.pi, np.pi), # np.arctan2(v[1], v[0]),
            'sensor_front': -1,
            'sensor_left': -1,
            'sensor_right': -1,
            'past_sensor_front': -1,
            'past_sensor_left': -1,
            'past_sensor_right': -1,
            'current_waypoint': 0,
            'next_waypoint': 1,
            'sensor_range': 30,
            'waypoints': waypoints,
            'space': space,
            'cars': cars,
            'walls': walls,
            'reward': 0,
            'cumulative_reward': 0,
            'reward_mean': 1,
            'reward_std': 1,
            'old_velocity': 0,
            'is_braking': False
        }
        self.speed_limit = 200
        self.steps_since_last_waypoint = 0
        self.steps_left = self.max_steps
        self.log = ''
        self.state_history = [self.state]
        self.reward_history = []
        self.car_color = np.random.randint(0, 255, 3)
        self.walls_draw = walls_draw

        return self.observation()

    def observation(self):
        return np.array([self.state[obs] for obs in self.observations])

    def step(self, action, render=True, training=True):
        ### Ideas
        # if doesn't move to next waypoint in <=100 steps, end episode
        # maybe a reset action to reset the car to the current waypoint (have to watch out for cheating)
        # Set of maps of varying difficulty: train agent until it can complete the easiest map, then move on to the next map
        #  - Maybe have a set of maps that are all the same difficulty, but have different shapes
        #  - If the agent can complete all of the maps, then move on to the next set of maps
        #  - If the agent starts to fail, then go back to the previous set of maps


        # Get actions
        steer, throttle = action
        if throttle < 0:
            self.state['is_braking'] = True
        else:
            self.state['is_braking'] = False
        car = self.state['cars'][0][0]

        a, b = self.boundaries[self.state['next_waypoint']]
        c, d = self.boundaries[self.state['next_waypoint'] + 1]
        a1, b1 = self.boundaries[self.state['current_waypoint']]
        c1, d1 = self.boundaries[self.state['current_waypoint'] + 1]
        inCurrentWaypoint = inPoly(car.position, [a1, b1, c1, d1])
        inNextWaypoint = inPoly(car.position, [a, b, c, d])

        reward = -0.01
        
        # Get new heading
        new_heading = getNewHeading(self.state['heading'], steer)
        self.state['heading'] = new_heading
        # Get new speed
        new_speed, collision_occured = getNewSpeed(self.state['speed'], throttle, self.speed_limit, car.velocity.length, self.state['old_velocity'])
        # If collision occured, penalty
        if collision_occured and not(inCurrentWaypoint) and not(inNextWaypoint):
            new_speed = new_speed * 0.1
            # Make sure by seeing if the car is not within the boundaries of the track
            # if reward > 0:
            #     reward -= 1000*len(self.state['waypoints'])
            self.reset_count += 1
        else:
            self.reset_count = 0

        self.state['speed'] = new_speed

        # Update car velocity
        car.velocity = pymunk.Vec2d(new_speed * np.cos(new_heading), new_speed * np.sin(new_heading))
        self.state['old_velocity'] = car.velocity.length

        # Update physics
        if render:
            self.render(training)
        else:
            self.state['space'].step(1 / FPS)

        # Get sensor readings
        s_range = self.state['sensor_range']
        sensor_front = sensorReading(0, car, self.state['space'], s_range+30, new_heading)
        sensor_left = sensorReading(-np.pi / 2, car, self.state['space'], s_range, new_heading)
        sensor_right = sensorReading(np.pi / 2, car, self.state['space'], s_range, new_heading)
        self.state['past_sensor_front'], self.state['past_sensor_left'], self.state['past_sensor_right'] = self.state['sensor_front'], self.state['sensor_left'], self.state['sensor_right']
        self.state['sensor_front'] = sensor_front
        self.state['sensor_left'] = sensor_left
        self.state['sensor_right'] = sensor_right

        # If in the next waypoint, update waypoints
        # if inRectangle(car.position, self.state['boundary_points'][self.state['next_waypoint']]):
        if inNextWaypoint:
            self.state['current_waypoint'] += 1
            self.state['next_waypoint'] += 1
            reward += 1000 / len(self.state['waypoints'])
            # reward += 2
            self.steps_since_last_waypoint = 0
        # Else if in the current waypoint, do nothing
        elif inCurrentWaypoint:
            reward += 0
            self.steps_since_last_waypoint += 1
        # Else if in neither waypoint, either the car is off the track or it is going backwards
        elif inPreviousWaypoints(self.boundaries, self.state['current_waypoint'], car.position):
            reward -= 0.1
            self.steps_since_last_waypoint += 1
            

        # Update state
        if (self.state['next_waypoint'] < len(self.state['waypoints']) - 1):
            v = self.state['vector_x'], self.state['vector_y'] = self.state['waypoints'][self.state['next_waypoint']] - car.position

        # Update steps left
        self.steps_left -= 1

        # Check if done
        done = any([
            # If the car has not reached the next waypoint in 100 steps, end the episode
            self.steps_since_last_waypoint >= 500,
            # If the car has reached the last waypoint, end the episode
            self.state['next_waypoint'] == len(self.state['waypoints']) - 1,
            # If the car has run out of steps, end the episode
            self.steps_left == 0,
            collision_occured and not(inCurrentWaypoint) and not(inNextWaypoint) and self.reset_count >= 5
        ])

        # if (done and self.steps_since_last_waypoint >= 500):
        #     reward -= 1

        if collision_occured or (done and self.steps_since_last_waypoint >= 500):
            # If collision occurred, set reward to 0 and remove cumulative reward
            if self.state['cumulative_reward'] > 0:
                # print('collision occured')
                reward = -self.state['cumulative_reward'] - reward

        self.state['reward'] = reward
        self.state['cumulative_reward'] += reward
        
        # State History
        self.state_history.append(self.state.copy())

        # Return observation, reward, done, info
        return self.observation(), reward, done, {}

    def render(self, training=True):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Update physics
        self.state['space'].step(1 / FPS)

        # Draw stuff
        car_position = self.state['cars'][0][0].position
        # Every 10 episodes, reset the screen
        if not(self.has_reset_screen) and self.num_episodes % 30 == 0:
            screen.fill((0, 0, 0))
            self.has_reset_screen = True
        if not(training):
            screen.fill((0, 0, 0))
            draw_walls(screen, self.state['walls'], self.walls_draw)
            draw_waypoints(screen, self.state['waypoints'], self.state['current_waypoint'], self.state['next_waypoint'])
            draw_timestep(screen, self.state_history)
            draw_sensors(screen, self.state)
            draw_speed(screen, self.state['cars'][0][0].velocity.length, self.speed_limit)
            draw_braking(screen, self.state['is_braking'])
            draw_reward(screen, self.state_history[-1]['reward'], self.state_history[-1]['cumulative_reward'])
        draw_cars(screen, self.state['cars'], self.car_color)
        draw_boundaries(screen, self.boundaries, self.state)
        # Update display (only update part of the screen that contains the car if training)
        if training:
            if self.num_episodes % 30 == 0:
                pygame.display.update()
            else:
                pygame.display.update((car_position[0] - 20, car_position[1] - 20, 40, 40))
            clock.tick(np.inf)
        else:
            pygame.display.update()
            clock.tick(FPS)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

### Game ###
def create_wall_radius(space, x1, y1, x2, y2, waypoint1, waypoint2):
    p1 = [x1, y1]
    p2 = [x2, y2]

    # Calculate vector from waypoint1 to p1
    v1 = np.array(p1) - np.array(waypoint1)
    # Calculate vector from waypoint2 to p2
    v2 = np.array(p2) - np.array(waypoint2)

    # Move the points along their respective vectors
    p1 = np.array(p1) + v1
    p2 = np.array(p2) + v2

    p1 = tuple(p1)
    p2 = tuple(p2)

    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, 0)
    segment = pymunk.Segment(body, p1, p2, 1)
    # Set the radius to the vector length
    segment.unsafe_set_radius(np.linalg.norm(v1))

    # Define elasticity
    segment.elasticity = 0.2
    space.add(body, segment)
    return segment

def create_wall_normal(space, x1, y1, x2, y2):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, 0)
    segment = pymunk.Segment(body, (x1, y1), (x2, y2), 3)
    # Define elasticity
    segment.elasticity = 0.8
    space.add(body, segment)
    return segment

def draw_walls(screen, walls, walls_draw):
    for wall in walls:
        pygame.draw.line(screen, (0, 255, 255), wall.a, wall.b)
    for wall in walls_draw:
        pygame.draw.line(screen, (255, 255, 255), wall[0], wall[1])

def create_car(space, x, y):
    body = pymunk.Body(10, 1, body_type=pymunk.Body.DYNAMIC)
    # Set mass to be at center of car
    body.position = (x, y)
    body.angular_velocity = np.pi / 2
    shape = pymunk.Circle(body, 2)
    shape.elasticity = 1
    space.add(body, shape)
    return body, shape

def draw_cars(screen, cars, color):
    for car in cars:
        pos = car[0].position
        pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 1)

def draw_waypoints(screen, waypoints, current_waypoint, next_waypoint):
    for i, waypoint in enumerate(waypoints):
        # Make the current waypoint red
        if i == current_waypoint:
            color = (255, 0, 0)
        # Make the next waypoint green
        elif i == next_waypoint:
            color = (0, 255, 0)
        # Make completed waypoints blue
        elif i < current_waypoint:
            color = (0, 0, 255)
        # Make future waypoints white
        else:
            color = (255, 255, 255)
        pygame.draw.circle(screen, color, (int(waypoint[0]), int(waypoint[1])), 2)

def draw_timestep(screen, state_history):
    # Write timestep to screen
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render(f'Timestep: {len(state_history)}', True, (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (500, 50)
    screen.blit(text, textRect)

def draw_sensors(screen, state):
    # Draw front sensor
    car = state['cars'][0][0]
    sensor_front = car.position + pymunk.Vec2d((state['sensor_range']+30) * np.cos(state['heading']), (state['sensor_range']+30) * np.sin(state['heading']))
    pygame.draw.line(screen, (255, 255, 255), car.position, sensor_front, 1)
    # Draw left sensor
    sensor_left = car.position + pymunk.Vec2d(state['sensor_range'] * np.cos(state['heading'] + np.pi / 2), state['sensor_range'] * np.sin(state['heading'] + np.pi / 2))
    pygame.draw.line(screen, (255, 255, 255), car.position, sensor_left, 1)
    # Draw right sensor
    sensor_right = car.position + pymunk.Vec2d(state['sensor_range'] * np.cos(state['heading'] - np.pi / 2), state['sensor_range'] * np.sin(state['heading'] - np.pi / 2))
    pygame.draw.line(screen, (255, 255, 255), car.position, sensor_right, 1)

def draw_reward(screen, reward, cumulative_reward):
    # # Write reward to screen
    # font = pygame.font.Font('freesansbold.ttf', 32)
    # text = font.render(f'Reward: {reward:.2f}', True, (255, 255, 255))
    # textRect = text.get_rect()
    # textRect.center = (500, 100)
    # screen.blit(text, textRect)
    # Write cumulative reward to screen
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render(f'Cumulative Reward: {cumulative_reward:.2f}', True, (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (500, 150)
    screen.blit(text, textRect)

def draw_speed(screen, velocity, speed_limit):
    # Write velocity to screen
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render(f'Speed: {velocity:.2f}', True, (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (500, 100)
    screen.blit(text, textRect)
    # Create speed bar
    pygame.draw.rect(screen, (255, 255, 255), (400, 150, speed_limit, 10))
    pygame.draw.rect(screen, (0, 255, 0), (400, 150, velocity, 10))

def draw_boundaries(screen, boundaries, state):
    # Get points a, b, c, d
    a, b = boundaries[state['current_waypoint']]
    c, d = boundaries[state['current_waypoint'] + 1]
    a1, b1 = boundaries[state['next_waypoint']]
    c1, d1 = boundaries[state['next_waypoint'] + 1]

    # Draw lines between points a-b, b-d, d-c, c-a
    pygame.draw.line(screen, (255, 255, 255), a, b, 1)
    pygame.draw.line(screen, (255, 255, 255), b, d, 1)
    pygame.draw.line(screen, (255, 255, 255), d, c, 1)
    pygame.draw.line(screen, (255, 255, 255), c, a, 1)

    pygame.draw.line(screen, (255, 255, 255), a1, b1, 1)
    pygame.draw.line(screen, (255, 255, 255), b1, d1, 1)
    pygame.draw.line(screen, (255, 255, 255), d1, c1, 1)
    pygame.draw.line(screen, (255, 255, 255), c1, a1, 1)

def draw_braking(screen, is_braking):
    if is_braking:
        # Write braking to screen
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(f'Braking', True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (500, 200)
        screen.blit(text, textRect)

def game():
    # Collision handler
    def car_collide(arbiter, space, data):
        # Bounce off walls
        return True
    
    # Add collision handler
    # When two cars collide, call car_collide\
    handler = space.add_collision_handler(0, 0)
    handler.begin = car_collide

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Update physics
        space.step(1 / FPS)

        # Draw stuff
        screen.fill((0, 0, 0))
        draw_walls(screen, walls)
        draw_cars(screen, cars)
        screen.blit(pygame.transform.flip(screen, False, True), (0, 0))
        pygame.display.update()
        clock.tick(FPS)

def createGame(space, boundaries, points):
    # Create car
    cars = []
    v = points[1] - points[0]
    # Start the car a little bit away from the first waypoint
    cars.append(create_car(space, points[0][0] + v[0] * 0.1, points[0][1] + v[1] * 0.1))
    # Add ShapeFilter to car
    cars[0][1].filter = pymunk.ShapeFilter(categories=0b1, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1)

    # Create simple line segments
    walls = []
    walls_draw = []
    for i in range(len(boundaries) - 1):
        # Create segment from point i to point i + 1
        p1 = boundaries[i][0]
        p2 = boundaries[i + 1][0]
        p3 = boundaries[i][1]
        p4 = boundaries[i + 1][1]
        walls.append(create_wall_radius(space, p1[0], p1[1], p2[0], p2[1], points[i], points[i + 1]))
        walls.append(create_wall_radius(space, p3[0], p3[1], p4[0], p4[1], points[i], points[i + 1]))
        walls_draw.append((p1, p2))
        walls_draw.append((p3, p4))
        # Create segment between p1 and p3
        if i == 0:
            walls.append(create_wall_radius(space, p1[0], p1[1], p3[0], p3[1], p2, p4))
        if i == len(boundaries) - 2:
            walls.append(create_wall_radius(space, p2[0], p2[1], p4[0], p4[1], p1, p3))

    # Collision handler
    def car_collide(arbiter, space, data):
        # print("Collision")
        # Bounce off walls
        return True
    
    def car_collide_end(arbiter, space, data):
        # print("Collision end")
        # Bounce off walls
        return True
    
    # Add collision handler
    handler = space.add_collision_handler(0, 0)
    handler.begin = car_collide
    handler.separate = car_collide_end

    return space, cars, walls, walls_draw

### Display Episodes in PyGame ###
def playNEpisodes(n, env, model):
    for episode in range(n):
        obs = env.reset()
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, render=True, training=False)
            if done:
                print(f'Episode {episode} finished after {step} steps')
                # Pause the game
                unpause = not(episode == n - 1)
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                unpause = True
                                break
                    if unpause:
                        break
                    pygame.display.update()
                break

### Parallel Environments ###
def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    def _init() -> gym.Env:
        env = CustomRacing2DEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def trainParallel():
    n_cpu = 8
    env = make_vec_env(CustomRacing2DEnv, n_envs=n_cpu, seed=0)
    return env

if __name__ == "__main__":
    # Create environment
    env = CustomRacing2DEnv()
    # env = trainParallel()
    # env = DummyVecEnv([lambda: CustomRacing2DEnv()])
    # env = VecNormalize(env)

    # Create model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train model
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save('../models/temp_model')

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

    # Render n episodes
    env = CustomRacing2DEnv()
    playNEpisodes(n, env, model)