'''
Cleaner version of the previous Boundary environment with some modifications to the observation space such that
the agent learns to operate from a first-person perspective (i.e. the agent considers the waypoints in front of it with respect to its
own position and orientation).
'''

### Imports ###
# Standard
import numpy as np
import matplotlib.pyplot as plt

# Gym
import gym

# SB3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Custom (other scripts)
from read_gpx import read_gpx, removeDuplicatePoints, scaleData

### Miscellaneous Functions ###
def boundaries(x, y):
    '''
    Returns the boundaries of the environment as a list of 4 points
    '''
    v = y - x
    p1 = np.array([v[1], -v[0]])
    p2 = np.array([-v[1], v[0]])

    p1 = p1 / np.linalg.norm(p1) * 0.05
    p2 = p2 / np.linalg.norm(p2) * 0.05

    p3 = p1 + v
    p4 = p2 + v

    points = np.array([p1, p2, p3, p4])
    points += x
    return points

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

def setPoints(data, d):
    '''
    Takes the scaled data points and returns a list of n equally spaced points such that each point is at least 'd' distance away from each other.
    '''
    # Initialize the list of points
    points = []
    # Initialize the first point
    points.append(data[0])
    # Initialize the current point
    current_point = data[0]
    # Initialize the current distance
    current_distance = 0
    # Initialize the index
    i = 1
    # Loop through the data
    while (i < len(data)):
        # Get the current distance
        current_distance = np.linalg.norm(data[i] - current_point)
        # If the current distance is greater than or equal to d
        if (current_distance >= d):
            # Add the current point to the list of points
            points.append(data[i])
            # Update the current point
            current_point = data[i]
        # Increment the index
        i += 1
    # Return the list of points
    return np.array(points)

def getRandomWaypoints(data):
    # Sample random starting point in the data
    start_index = np.random.randint(0, len(data) - 200)
    return np.array([[x[0], x[1]] for x in scaleData(data[start_index:start_index+100])])

### Global Variables ###
max_steps = 1000
max_episodes = 1000
n = 1
data_range = 100
num_waypoints = data_range
d = 0.1
total_timesteps = 200000
gamma = 0.99
file_name = "Evening_Ride.gpx"

reward_parameters = [10, 100]

waypoints = np.array([[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5]])
data = removeDuplicatePoints(read_gpx("../gpx/" + file_name))
waypoints = np.array([[x[0], x[1]] for x in setPoints(scaleData(removeDuplicatePoints(read_gpx("../gpx/Evening_Ride.gpx"))[500:500+data_range]), d)])
# waypoints = np.array([[x[0], x[1]] for x in setPoints(scaleData(removeDuplicatePoints(read_gpx("../gpx/" + file_name, 1))), d)])
boundary_points = np.array([boundaries(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1)])

### Spaces ###
# Observation Space
'''
1/2. Vector to next waypoint (agent's position - next waypoint's position)
3. Distance to next waypoint (scalar)
4. Speed (scalar)
5. Heading (angle between agent's heading and vector to next waypoint)
'''
observations = ['vector_x', 'vector_y', 'speed', 'heading', 'forward_sensor', 'left_sensor', 'right_sensor']

def defineObservationSpace():
    '''
    Defines the observation space based on the observations list.
    '''
    lows = {
        'vector_x': -np.inf,
        'vector_y': -np.inf,
        'speed': 0,
        'heading': -np.pi,
        'forward_sensor': 0,
        'left_sensor': 0,
        'right_sensor': 0
    }
    uppers = {
        'vector_x': np.inf,
        'vector_y': np.inf,
        'speed': np.inf,
        'heading': np.pi,
        'forward_sensor': 1,
        'left_sensor': 1,
        'right_sensor': 1
    }
    low = np.array([lows[obs] for obs in observations])
    high = np.array([uppers[obs] for obs in observations])
    return gym.spaces.Box(low, high, dtype=np.float32)

# Action Space
'''
1. Steering angle (scalar)
    1.1. Left (negative)
    1.2. Right (positive)
    1.3. Straight (0)
'''    
actions = ['steering_angle', 'speed_change']

def defineActionSpace():
    '''
    Defines the action space based on the actions list.
    '''
    lows = {
        'steering_angle': -np.pi / 8,
        'speed_change': -0.001
    }
    uppers = {
        'steering_angle': np.pi / 8,
        'speed_change': 0.001
    }
    low = np.array([lows[act] for act in actions])
    high = np.array([uppers[act] for act in actions])
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

def getNewPosition(state, steering_angle):
    '''
    Returns the new position of the agent based on the current vector, heading angle and speed.
    The heading angle is the angle between the agent's heading and the vector to the next waypoint.
    '''
    # Get the current vector
    vector = np.array([state['vector_x'], state['vector_y']])
    # Get the current heading angle
    heading_angle = state['heading']
    # Get the current speed
    speed = state['speed']
    # Get the current position
    position = state['position']

    # Translate the vector to the origin
    v = vector - position

    # Get the rotation matrix
    R = np.array([[np.cos(heading_angle), -np.sin(heading_angle)], [np.sin(heading_angle), np.cos(heading_angle)]])
    # Rotate the vector
    v = R @ v
    # Scale the vector to be equivalent to the speed
    v =  v / np.linalg.norm(v) * speed
    # Translate the vector back to the original position
    v += position  
    # Return the new position
    return v

def getNewSpeed(state, speed_change):
    '''
    Returns the new speed of the agent based on the current speed and speed change.
    '''
    # Get the current speed
    speed = state['speed']
    # Return the new speed (cap the speed at 0 and 0.01)
    return np.clip(speed + speed_change, 0, 0.01)

def checkForwardSensor(state):
    ''' NEEDS REFACTORING
    Returns 0 if the forward sensor does not detect anything.
    Returns 1 if the forward sensor detects something (sensor point is outside the boundaries given).
    '''
    # Get the current vector
    vector = np.array([state['vector_x'], state['vector_y']])
    # Get the current position
    position = state['position']
    # Get the current heading angle
    heading_angle = state['heading']

    ### Get the sensor point
    # Translate the vector to the origin
    v = vector - position
    # Get the rotation matrix
    R = np.array([[np.cos(heading_angle), -np.sin(heading_angle)], [np.sin(heading_angle), np.cos(heading_angle)]])
    # Rotate the vector
    v = R @ v
    # Scale the vector to be equivalent to some distance
    v =  v / np.linalg.norm(v) * 1
    # Translate the vector back to the original position
    v += position
    sensor_point = v

    ### Check if the sensor point is in the boundaries
    # Check the current and next waypoint's boundaries (i.e. it's okay to go into the next waypoint's boundary)
    if (inRectangle(sensor_point, state['boundary_points'][state['current_waypoint']]) or inRectangle(sensor_point, state['boundary_points'][state['next_waypoint']])):
        return 0
    else:
        return 1

def checkSensor(state, direction):
    ''' NEEDS REFACTORING
    Returns 0 if the forward sensor does not detect anything.
    Returns 1 if the forward sensor detects something (sensor point is outside the boundaries given).
    '''
    # Get the current vector
    vector = np.array([state['vector_x'], state['vector_y']])
    # Get the current position
    position = state['position']
    # Get the current heading angle
    heading_angle = state['heading']

    ### Get the sensor point
    # Translate the vector to the origin
    v = vector - position
    # Get the rotation matrix
    R = np.array([[np.cos(heading_angle+direction), -np.sin(heading_angle+direction)], [np.sin(heading_angle+direction), np.cos(heading_angle+direction)]])
    # Rotate the vector
    v = R @ v
    # Scale the vector to be equivalent to some distance
    v =  v / np.linalg.norm(v) * 1
    # Translate the vector back to the original position
    v += position
    sensor_point = v

    ### Check if the sensor point is in the boundaries
    # Check the current and next waypoint's boundaries (i.e. it's okay to go into the next waypoint's boundary)
    if (inRectangle(sensor_point, state['boundary_points'][state['current_waypoint']]) or inRectangle(sensor_point, state['boundary_points'][state['next_waypoint']])):
        return 0
    else:
        return 1


def rewardFunction(state_history, factorParameters):
    '''
    Returns the reward based on the state history.
    Factors:
    1. Ratio of waypoints reached
    2. Average speed
    '''
    waypointParameter = factorParameters[0]
    speedParameter = factorParameters[1]

    # Initialize the reward
    reward = 0
    # Initialize the average speed
    average_speed = 0
    # Initialize the number of steps
    num_steps = len(state_history)
    # Loop through the state history
    for i in range(num_steps - 1):
        # Get the current speed
        current_speed = state_history[i]['speed']
        # Increment the average speed
        average_speed += current_speed
    # Get the average speed
    average_speed /= num_steps
    # Get the number of waypoints reached
    num_waypoints_reached = state_history[-1]['current_waypoint'] + 1
    ratio_waypoints_reached = num_waypoints_reached / len(state_history[-1]['waypoints'])
    # Get the reward
    reward = (ratio_waypoints_reached * waypointParameter) + (average_speed * speedParameter)
    # Return the reward
    return reward

### Environment ###
class BoundaryV2Env(gym.Env):
    def __init__(self):
        self.observations = observations
        self.observation_space = defineObservationSpace()
        self.actions = actions
        self.action_space = defineActionSpace()
        self.max_steps = max_steps

    def reset(self):
        # Change the waypoints
        waypoints = getRandomWaypoints(data)
        boundary_points = np.array([boundaries(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1)])
        v = waypoints[1] - waypoints[0]
        self.state = {
            'vector_x': v[0],
            'vector_y': v[1],
            'distance': np.linalg.norm(v),
            'speed': 0.005,
            'heading': np.radians(0),
            'forward_sensor': 0,
            'left_sensor': 0,
            'right_sensor': 0,
            'current_waypoint': 0,
            'next_waypoint': 1,
            'position': np.array(waypoints[0]),
            'waypoints': waypoints,
            'boundary_points': boundary_points
        }
        self.steps_left = self.max_steps
        self.log = ''
        self.state_history = [self.state]

        return self.observation()

    def observation(self):
        return np.array([self.state[obs] for obs in self.observations])

    def step(self, action):
        # Get new heading
        new_heading = getNewHeading(self.state['heading'], action[0])
        self.state['heading'] = new_heading

        # Get new speed
        new_speed = getNewSpeed(self.state, action[1])
        self.state['speed'] = new_speed

        # Get new position
        new_position = getNewPosition(self.state, action[0])
        self.state['position'] = new_position

        # Get new sensor readings
        self.state['forward_sensor'] = checkSensor(self.state, 0)
        self.state['left_sensor'] = checkSensor(self.state, np.pi / 4)
        self.state['right_sensor'] = checkSensor(self.state, -np.pi / 4)

        # Initialize reward
        reward = 0

        ### Check which boundary the agent is in
        # If in the next waypoint's boundary
        if (inRectangle(new_position, self.state['boundary_points'][self.state['next_waypoint']])):
            # Increment the waypoint data
            self.state['current_waypoint'] += 1
            self.state['next_waypoint'] += 1

            num_waypoints_so_far = self.state['current_waypoint'] + 1

            # Increment the reward
            reward += 20 / len(self.state['waypoints'])
        # Else if still in the current waypoint's boundary
        elif (inRectangle(new_position, self.state['boundary_points'][self.state['current_waypoint']])):
            # Decrement the reward
            reward -= 0.01
        # Else if in neither of the boundaries
        else:
            # Decrement the reward
            reward -= 0.01
            pass

        # Update the state
        if (self.state['next_waypoint'] < len(self.state['waypoints']) - 1):
            v = self.state['vector_x'], self.state['vector_y'] = self.state['waypoints'][self.state['next_waypoint']] - self.state['position']
            self.state['distance'] = np.linalg.norm(v)

        # Copy the new state to the state history
        self.state_history.append(self.state.copy())

        # Update the step count
        self.steps_left -= 1

        # Check if the episode is done
        done = False
        if (self.steps_left == 0):
            done = True
            reward = -1
        elif (self.state['next_waypoint'] == len(self.state['waypoints']) - 1):
            done = True
            reward = 1

        # Additional reward considerations if the episode is done
        # if (done):
        #     reward += rewardFunction(self.state_history, reward_parameters)
        
        # Return the observation, reward, done and info
        return self.observation(), reward, done, {}

    def render(self, mode='log'):
        if (mode == 'log'):
            print(self.log)
            self.log = ''
        elif (mode == 'plot'):
            self.renderPath()
        else:
            pass

    def renderPath(self):
        # Plot the waypoints
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=2)
        # Plot the boundary points
        for i, boundary in enumerate(boundary_points):
            plt.plot(boundary[:, 0], boundary[:, 1], 'bo', markersize=0.5)
        # Plot the agent's path
        path = np.array([state['position'] for state in self.state_history])
        colors = np.linspace(0, 1, len(path))
        plt.scatter(path[:, 0], path[:, 1], c=colors, cmap='viridis')

        # Plot settings
        plt.grid()  
        plt.xlim(-0.5, 2.0)
        plt.ylim(-0.5, 2.0)
        plt.show()
        
    def close(self):
        pass

### Plotting Functions ###
def plotNEpisodes(n):
    path_trajectories = []
    reward_hist = []
    for episode in range(n):
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            reward_hist.append(rewards)
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                # print("Final State:", env.state)
                path_trajectories.append(env.state_history)
                break
        # env.render(mode='path')
    
    # Plot the trajectories
    for trajectory in path_trajectories:
        path = np.array([state['position'] for state in trajectory])
        colors = np.linspace(0, 1, len(path))
        plt.scatter(path[:,0], path[:,1], c=colors, cmap='cool', s=0.3)
        # # Label all the positive rewards
        # for i, point in enumerate(path[:-1]):
        #     if (reward_hist[i] > 0):
        #         plt.text(point[0], point[1], reward_hist[i])
    # Plot the waypoints
    plt.plot(waypoints[:,0], waypoints[:,1], 'ro', markersize=2)
    # Label the waypoints
    for i, waypoint in enumerate(waypoints):
        plt.text(waypoint[0], waypoint[1], str(i+1))
    # Plot the boundary points and lines
    for i, bp in enumerate(boundary_points):
        # Points
        plt.plot(bp[:,0], bp[:,1], 'bo', markersize=0.5)
        # Lines
        # 1 and 2
        plt.plot([bp[0][0], bp[1][0]], [bp[0][1], bp[1][1]], c='black')
        # 2 and 4
        # plt.plot([bp[1][0], bp[3][0]], [bp[1][1], bp[3][1]], c='black')
        # 4 and 3
        # plt.plot([bp[3][0], bp[2][0]], [bp[3][1], bp[2][1]], c='black')
        # 3 and 1
        # plt.plot([bp[2][0], bp[0][0]], [bp[2][1], bp[0][1]], c='black') 
    plt.grid()  
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()

if __name__ == "__main__":
    # Create the environment
    env = BoundaryV2Env()

    # Create the model
    model = PPO('MlpPolicy', env, verbose=1, gamma=gamma)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save('ppo_boundary_v2')

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Print the mean and std reward
    print(f'Mean Reward: {mean_reward} | Std Reward: {std_reward}')

    # Render the path
    env.render()

    # Render n episodes
    plotNEpisodes(n)

    # Print log
    # env.render(mode='log')