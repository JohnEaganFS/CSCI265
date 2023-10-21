'''
Very similar to the previous environment, but I want the agent to now have constraints on where they can go in the environment
according to path points and their boundaries. This will essentially be an A to B with "walls" that the agent cannot go through.
'''

# Imports
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from read_gpx import read_gpx, removeDuplicatePoints, scaleData


# Parameters
max_steps = 500
# waypoints = np.array([[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5]])
# waypoints = np.array([[0, 0], [0.5, 0.75], [1, 1], [1.5, 1.5]])
waypoints = np.array([[0,0], [0, 0.25], [0, 0.5], [0.5, 0.75], [1, 1], [1.5, 1.5]])

# Observations
observations = ['lat', 'lon', 'speed', 'heading_lat', 'heading_lon', 'next_waypoint_lat', 'next_waypoint_lon']

def defineObservationSpace():
    lower_bounds = {
        'lat': 0,
        'lon': 0,
        'speed': 0,
        'heading_lat': 0,
        'heading_lon': 0,
        'next_waypoint_lat': -np.inf,
        'next_waypoint_lon': -np.inf
    }
    upper_bounds = {
        'lat': 1,
        'lon': 1,
        'speed': 1,
        'heading_lat': 1,
        'heading_lon': 1,
        'next_waypoint_lat': np.inf,
        'next_waypoint_lon': np.inf
    }
    low = np.array([lower_bounds[key] for key in observations])
    high = np.array([upper_bounds[key] for key in observations])
    shape = low.shape
    return gym.spaces.Box(low, high, shape, dtype=np.float32)
    
# Actions
def getNewHeading(state, action):
    # Get the agent's current position
    position = np.array([state['lat'], state['lon']])

    # Get the heading vector
    heading = np.array([state['heading_lat'], state['heading_lon']])

    # Translate heading vector to the origin
    heading = heading - position

    # Rotate the heading vector some random amount between 5-10 degrees to the left
    angle = np.radians(np.random.uniform(5, 10))

    # Create the rotation matrix
    if (action == 'left'):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    elif (action == 'right'):
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    # Rotate the heading vector
    heading = np.matmul(rotation_matrix, heading)

    # Translate the heading vector back to the agent's position
    heading = heading + position

    return heading

def left(state):
    state['heading_lat'], state['heading_lon'] = getNewHeading(state, 'left')

def right(state):
    state['heading_lat'], state['heading_lon'] = getNewHeading(state, 'right')

def straight(state):
    pass

actions = [left, right, straight]

def defineActionSpace():
    return gym.spaces.Discrete(len(actions))

def updateSpeed(state):
    pass

# Rewards

# Functions
def boundaries(a, b):
    # get a and b if translated such that a is at the origin
    vector = b - a

    point1 = np.array([vector[1], -vector[0]])
    point2 = np.array([-vector[1], vector[0]])

    # scale point1 and point2 to be 0.1 units long
    point1 = point1 / np.linalg.norm(point1) * 0.2
    point2 = point2 / np.linalg.norm(point2) * 0.2

    point3 = point1 + vector
    point4 = point2 + vector

    # translate back to original position
    point1 = point1 + a
    point2 = point2 + a
    point3 = point3 + a
    point4 = point4 + a

    # Return the four points
    return [point1, point2, point3, point4]

def checkIfInRectangle(point, boundary_points):
    ''' Check if a point is in a rectangle defined by the boundary points.
    '''
    # Get the boundary points
    a, b, d, c = boundary_points[0], boundary_points[1], boundary_points[2], boundary_points[3]
    # Check if point is in rectangle
    vector_ap = point - a
    vector_ab = b - a
    vector_ad = d - a
    return 0 <= np.dot(vector_ap, vector_ab) <= np.dot(vector_ab, vector_ab) and 0 <= np.dot(vector_ap, vector_ad) <= np.dot(vector_ad, vector_ad)

# Environment
class BoundaryEnv(gym.Env):
    def __init__(self):
        self.observations = observations
        self.observation_space = defineObservationSpace()
        self.actions = actions
        self.action_space = defineActionSpace()
        self.max_steps = max_steps

    def observation(self):
        return np.array([self.state[key] for key in self.observations])

    def step(self, action):
        # Take the action
        self.actions[action](self.state)

        # Update the agent's position
        dx = self.state['heading_lat'] - self.state['lat']
        dy = self.state['heading_lon'] - self.state['lon']

        reward = 0

        # Check if the agent is within the current boundary
        # If it is, proceed as normal
        if (checkIfInRectangle(np.array([self.state['lat'], self.state['lon']]), self.state['boundary_points'][self.state['current_waypoint_index']])):
            self.state['lat'] += self.state['speed'] * dx
            self.state['lon'] += self.state['speed'] * dy
            self.state['heading_lat'] += self.state['speed'] * dx
            self.state['heading_lon'] += self.state['speed'] * dy
            reward = 0
        # If it isn't, check if it is within the next boundary
        # If it is, update the current and next waypoint indices
        elif (checkIfInRectangle(np.array([self.state['lat'], self.state['lon']]), self.state['boundary_points'][self.state['next_waypoint_index']])):
            self.state['current_waypoint_index'] += 1
            self.state['next_waypoint_index'] += 1
            self.state['lat'] += self.state['speed'] * dx
            self.state['lon'] += self.state['speed'] * dy
            self.state['heading_lat'] += self.state['speed'] * dx
            self.state['heading_lon'] += self.state['speed'] * dy
            self.state['next_waypoint_lat'] = self.state['waypoints'][self.state['next_waypoint_index']][0]
            self.state['next_waypoint_lon'] = self.state['waypoints'][self.state['next_waypoint_index']][1]
            reward = 20
            # Log these positive rewards
            self.log += f'Positive Reward: {self.state}\n'
        # If it isn't, then the agent is outside the boundaries
        else:
            # Move the agent back to the previous waypoint
            # self.state['lat'] = waypoints[self.state['current_waypoint_index']][0]
            # self.state['lon'] = waypoints[self.state['current_waypoint_index']][1]
            # # Reset the heading vector to the next waypoint
            # self.state['heading_lat'] = waypoints[self.state['next_waypoint_index']][0]
            # self.state['heading_lon'] = waypoints[self.state['next_waypoint_index']][1]
            self.state['lat'] += self.state['speed'] * dx
            self.state['lon'] += self.state['speed'] * dy
            self.state['heading_lat'] += self.state['speed'] * dx
            self.state['heading_lon'] += self.state['speed'] * dy

            # Reset penalty
            reward = -1

        # Update the speed
        updateSpeed(self.state)

        # Apply living penalty
        # reward -= 0.1
        
        # Copy the state to the state history
        self.state_history.append(self.state.copy())

        # Update the number of steps left
        self.steps_left -= 1

        # Check if the episode is over
        done = False
        if self.steps_left == 0:
            done = True
        elif self.state['current_waypoint_index'] == len(self.state['waypoints']) - 2:
            done = True
            reward = 1000

        # Return
        return self.observation(), reward, done, {}
        
    def reset(self):
        self.state = {
            'lat': waypoints[0][0],
            'lon': waypoints[0][1],
            'speed': 0.05,
            'heading_lat': waypoints[1][0],
            'heading_lon': waypoints[1][1],
            'next_waypoint_lat': waypoints[1][0],
            'next_waypoint_lon': waypoints[1][1],
            'current_waypoint_index': 0,
            'next_waypoint_index': 1,
            'waypoints': waypoints,
            'boundary_points': boundary_points,
            'best_distance': np.inf
        }
        self.steps_left = self.max_steps
        self.log = f'Initial State: {self.state}\n'
        self.state_history = [self.state]

        return self.observation()

    def render(self, mode='human'):
        # print(self.log)
        self.log = ''

        if mode == 'path':
            print("Rendering path")
            self.renderPath()
        
    def renderPath(self):
        # Plot the agent's path
        path = np.array([[state['lat'], state['lon']] for state in self.state_history])
        colors = np.linspace(0, 1, len(path))
        plt.scatter(path[:,0], path[:,1], c=colors, cmap='cool', s=0.1)
        plt.xlim = (0, 1)
        plt.ylim = (0, 1)
        plt.grid()
        # Plot the waypoints
        plt.plot(waypoints[:,0], waypoints[:,1], 'ro')
        # Plot the boundary points and lines
        for i, bp in enumerate(boundary_points):
            # Points
            plt.plot(bp[:,0], bp[:,1], 'bo')
            # Lines
            # 1 and 2
            plt.plot([bp[0][0], bp[1][0]], [bp[0][1], bp[1][1]], c='black')
            # 2 and 4
            plt.plot([bp[1][0], bp[3][0]], [bp[1][1], bp[3][1]], c='black')
            # 4 and 3
            plt.plot([bp[3][0], bp[2][0]], [bp[3][1], bp[2][1]], c='black')
            # 3 and 1
            plt.plot([bp[2][0], bp[0][0]], [bp[2][1], bp[0][1]], c='black')
            
        
        plt.show()

    def close(self):
        pass

def one_step(env):
    # Select an action
    action = env.action_space.sample()

    # Take the action
    observation, reward, done, info = env.step(action)

if __name__ == "__main__":
    data = read_gpx("../gpx/Evening_Ride.gpx")

    # Remove duplicate points
    data = removeDuplicatePoints(data)

    data = data[100:110]

    # Scale the lat/lon and ele coordinates to be in the range [0, 1]
    data = scaleData(data)

    # Convert data to waypoints
    waypoints = np.array([[x[0], x[1]] for x in data])

    # Create the environment
    env = BoundaryEnv()
    boundary_points = np.array([boundaries(waypoints[i], waypoints[i+1]) for i in range(len(waypoints) - 1)])

    # # Reset the environment
    # observation = env.reset()

    # # Run the environment
    # steps = 10

    # for step in range(steps):
    #     one_step(env)

    # # Render final environment
    # env.render(mode='path')

    # Model
    model = PPO('MlpPolicy', env, verbose=1)
    # model = DQN('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=300000)

    # Render
    env.render()

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Render n episodes of the model
    n = 1
    path_trajectories = []
    for episode in range(n):
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if done:
                # print("Episode finished after {} timesteps".format(i+1))
                # print("Final State:", env.state)
                path_trajectories.append(env.state_history)
                break
        # env.render(mode='path')
    
    # Plot the trajectories
    for trajectory in path_trajectories:
        path = np.array([[state['lat'], state['lon']] for state in trajectory])
        colors = np.linspace(0, 1, len(path))
        plt.scatter(path[:,0], path[:,1], c=colors, cmap='cool', s=0.1)
    # Plot the waypoints
    plt.plot(waypoints[:,0], waypoints[:,1], 'ro', markersize=2)
    # Plot the boundary points and lines
    for i, bp in enumerate(boundary_points):
        # Points
        plt.plot(bp[:,0], bp[:,1], 'bo', markersize=0.5)
        # Lines
        # 1 and 2
        # plt.plot([bp[0][0], bp[1][0]], [bp[0][1], bp[1][1]], c='black')
        # 2 and 4
        #plt.plot([bp[1][0], bp[3][0]], [bp[1][1], bp[3][1]], c='black')
        # 4 and 3
        #plt.plot([bp[3][0], bp[2][0]], [bp[3][1], bp[2][1]], c='black')
        # 3 and 1
        #plt.plot([bp[2][0], bp[0][0]], [bp[2][1], bp[0][1]], c='black') 
    plt.grid()  
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()