'''
Switching to a 2D physics engine library for the sake of simplicity. I will be using PyMunk. This will enable be me to
create the "walls" of the track and the "car" that will be racing around the track.
'''

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
FPS = 60

# Pull waypoints from .gpx file
points = removeDuplicatePoints(read_gpx('../gpx/windy_road.gpx', 1))
points = points[10:30]
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

min_vector_length = min([np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)]) * 2

# Create boundaries
boundary_points = define_boundaries(points, min_vector_length)
# print(boundary_points)


### Observations ###
observations = ['vector_x', 'vector_y', 'speed', 'heading', 'sensor_front', 'sensor_left', 'sensor_right']

def defineObservationSpace():
    lows = {
        'vector_x': -np.inf,
        'vector_y': -np.inf,
        'speed': 0,
        'heading': -np.pi,
        'sensor_front': 0,
        'sensor_left': 0,
        'sensor_right': 0
    }
    highs = {
        'vector_x': np.inf,
        'vector_y': np.inf,
        'speed': 10,
        'heading': np.pi,
        'sensor_front': np.inf,
        'sensor_left': np.inf,
        'sensor_right': np.inf
    }
    low = np.array([lows[obs] for obs in observations])
    high = np.array([highs[obs] for obs in observations])
    return gym.spaces.Box(low, high, dtype=np.float32)

### Actions ###
actions = ['steer', 'throttle', 'brake']

def defineActionSpace():
    lows = {
        'steer': -np.pi / 8,
        'throttle': 0,
        'brake': 0
    }
    highs = {
        'steer': np.pi / 8,
        'throttle': 1,
        'brake': 1
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
    steering_angle += np.random.normal(0, 0.05)
    new_angle = heading_angle - steering_angle
    # Keep the angle between -pi and pi
    if (new_angle > np.pi):
        new_angle -= 2 * np.pi
    elif (new_angle < -np.pi):
        new_angle += 2 * np.pi
    return new_angle

### Environment ###
class CustomRacing2DEnv(gym.Env):
    def __init__(self):
        self.observations = observations
        self.observation_space = defineObservationSpace()
        self.actions = actions
        self.action_space = defineActionSpace()
        self.max_steps = max_steps

    def reset(self):
        waypoints = points
        full_boundary_points = np.array([boundaries(waypoints[i], waypoints[i + 1], min_vector_length) for i in range(len(waypoints) - 1)])

        # Create new game
        space = pymunk.Space()
        cars, walls = createGame(space)

        v = waypoints[1] - waypoints[0]
        self.state = {
            'vector_x': v[0],
            'vector_y': v[1],
            'speed': cars[0][0].velocity.length,
            'heading': np.arctan2(v[1], v[0]),
            'sensor_front': 0,
            'sensor_left': 0,
            'sensor_right': 0,
            'current_waypoint': 0,
            'next_waypoint': 1,
            'waypoints': waypoints,
            'boundary_points': full_boundary_points,
            'space': space,
            'cars': cars,
            'walls': walls
        }
        self.steps_left = self.max_steps
        self.log = ''
        self.state_history = [self.state]

        return self.observation()

    def observation(self):
        return np.array([self.state[obs] for obs in self.observations])

    def step(self, action, render=False):
        # Get actions
        steer, throttle, brake = action
        
        # Get new heading
        new_heading = getNewHeading(self.state['heading'], steer)
        self.state['heading'] = new_heading
        # Get new speed
        new_speed = self.state['speed'] + throttle - brake 
        self.state['speed'] = new_speed

        # Simulate physics with pymunk space
        # Apply force to car
        force = pymunk.Vec2d(new_speed * np.cos(new_heading), new_speed * np.sin(new_heading))
        car = self.state['cars'][0][0]
        car.velocity = force
        # Update physics
        if render:
            self.render()
        else:
            self.state['space'].step(1 / FPS)

        # Get sensor readings
        sensor_front = car.position + pymunk.Vec2d(5 * np.cos(new_heading), 5 * np.sin(new_heading))
        sensor_left = car.position + pymunk.Vec2d(5 * np.cos(new_heading + np.pi / 2), 5 * np.sin(new_heading + np.pi / 2))
        sensor_right = car.position + pymunk.Vec2d(5 * np.cos(new_heading - np.pi / 2), 5 * np.sin(new_heading - np.pi / 2))
        # Check if sensor hits a wall
        front_hit = self.state['space'].segment_query_first(sensor_front, car.position, 1, pymunk.ShapeFilter())
        left_hit = self.state['space'].segment_query_first(sensor_left, car.position, 1, pymunk.ShapeFilter())
        right_hit = self.state['space'].segment_query_first(sensor_right, car.position, 1, pymunk.ShapeFilter())
        # If sensor hits a wall, get the point of intersection
        if front_hit:
            sensor_front = front_hit.point
        if left_hit:
            sensor_left = left_hit.point
        if right_hit:
            sensor_right = right_hit.point
        # Calculate distance from car to wall
        sensor_front = np.linalg.norm(sensor_front - car.position)
        sensor_left = np.linalg.norm(sensor_left - car.position)
        sensor_right = np.linalg.norm(sensor_right - car.position)
        # Update state
        self.state['sensor_front'] = sensor_front
        self.state['sensor_left'] = sensor_left
        self.state['sensor_right'] = sensor_right

        reward = -0.01

        # If in the next waypoint, update waypoints
        # if inRectangle(car.position, self.state['boundary_points'][self.state['next_waypoint']]):
        a, b = boundary_points[self.state['next_waypoint']]
        c, d = boundary_points[self.state['next_waypoint'] + 1]
        if inPoly(car.position, [a, b, c, d]):
            self.state['current_waypoint'] += 1
            self.state['next_waypoint'] += 1
            reward += 1000 / len(self.state['waypoints'])
        # Else if in the current waypoint, do nothing
        elif inRectangle(car.position, self.state['boundary_points'][self.state['current_waypoint']]):
            reward += 0
        # Else if in neither waypoint,
        else:
            reward -= 0.01

        # Update state
        if (self.state['next_waypoint'] < len(self.state['waypoints']) - 1):
            v = self.state['vector_x'], self.state['vector_y'] = self.state['waypoints'][self.state['next_waypoint']] - car.position
        
        # State History
        self.state_history.append(self.state.copy())

        # Update steps left
        self.steps_left -= 1

        # Check if done
        done = False
        if self.steps_left == 0:
            done = True
        elif self.state['next_waypoint'] == len(self.state['waypoints']) - 1:
            done = True
        
        # Return observation, reward, done, info
        return self.observation(), reward, done, {}

    def render(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Update physics
        self.state['space'].step(1 / FPS)

        # Draw stuff
        screen.fill((0, 0, 0))
        draw_walls(screen, self.state['walls'])
        draw_cars(screen, self.state['cars'])
        draw_waypoints(screen, self.state['waypoints'], self.state['current_waypoint'], self.state['next_waypoint'])
        draw_timestep(screen, self.state_history)
        # screen.blit(pygame.transform.flip(screen, False, True), (0, 0))
        pygame.display.update()
        clock.tick(FPS)

    def close(self):
        pass

### Game ###
def create_wall(space, x1, y1, x2, y2):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, 0)
    segment = pymunk.Segment(body, (x1, y1), (x2, y2), 1)
    segment.elasticity = 1
    space.add(body, segment)
    return segment

def draw_walls(screen, walls):
    for wall in walls:
        pygame.draw.line(screen, (255, 255, 255), wall.a, wall.b, 1)

def create_car(space, x, y):
    body = pymunk.Body(10, 1, body_type=pymunk.Body.DYNAMIC)
    # Set mass to be at center of car
    body.position = (x, y)
    body.angular_velocity = np.pi / 2
    shape = pymunk.Circle(body, 2)
    shape.elasticity = 1
    space.add(body, shape)
    return body, shape

def draw_cars(screen, cars):
    for car in cars:
        pos = car[0].position
        pygame.draw.circle(screen, (255, 0, 0), (int(pos[0]), int(pos[1])), 2)

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

def createGame(space):
    # Create car
    cars = []
    cars.append(create_car(space, points[0][0], points[0][1]))

    # Create simple line segments
    walls = []
    for i in range(len(boundary_points) - 1):
        # Create segment from point i to point i + 1
        p1 = boundary_points[i][0]
        p2 = boundary_points[i + 1][0]
        p3 = boundary_points[i][1]
        p4 = boundary_points[i + 1][1]
        walls.append(create_wall(space, p1[0], p1[1], p2[0], p2[1]))
        walls.append(create_wall(space, p3[0], p3[1], p4[0], p4[1]))
        # Create segment between p1 and p3
        # walls.append(create_wall(space, p1[0], p1[1], p3[0], p3[1]))

    # Collision handler
    def car_collide(arbiter, space, data):
        # Bounce off walls
        return True
    
    # Add collision handler
    # When two cars collide, call car_collide\
    handler = space.add_collision_handler(0, 0)
    handler.begin = car_collide

    return cars, walls

### Display Episodes in PyGame ###
def playNEpisodes(n, env, model):
    for episode in range(n):
        obs = env.reset()
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, render=True)
            if done:
                print(f'Episode {episode} finished after {step} steps')
                break

if __name__ == "__main__":

    # Create car
    # cars = []
    # cars.append(create_car(space, points[0][0], points[0][1]))

    # # Run game
    # game()

    # Create environment
    env = CustomRacing2DEnv()

    # Create model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train model
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save('ppo_custom_racing_2d')

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

    # Render n episodes
    playNEpisodes(n)