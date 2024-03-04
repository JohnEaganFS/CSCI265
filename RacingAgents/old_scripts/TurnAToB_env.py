'''
This script will create a gym environment for the agents to interact with.
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Parameters
max_steps = 1000

# Observations
observations = ['lat', 'lon', 'speed', 'heading_lat', 'heading_lon']

def defineObservationSpace():
    lower_bounds = {
        'lat': 0,
        'lon': 0,
        'speed': 0,
        'heading_lat': 0,
        'heading_lon': 0
    }
    upper_bounds = {
        'lat': 1,
        'lon': 1,
        'speed': 1,
        'heading_lat': 1,
        'heading_lon': 1
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

# Environment
class CyclingEnv(gym.Env):
    def __init__(self):
        self.observations = observations
        self.observation_space = defineObservationSpace()
        self.actions = actions
        self.action_space = defineActionSpace()
        self.max_steps = max_steps

    def step(self, action):
        # Take the action
        self.actions[action](self.state)

        # Update the agent's position
        dx = self.state['heading_lat'] - self.state['lat']
        dy = self.state['heading_lon'] - self.state['lon']
        self.state['lat'] += self.state['speed'] * dx
        self.state['lon'] += self.state['speed'] * dy

        # Update the heading vector
        self.state['heading_lat'] += self.state['speed'] * dx
        self.state['heading_lon'] += self.state['speed'] * dy

        # Update the speed
        updateSpeed(self.state)

        # Calculate the reward (did we get closer to the goal?)
        goal = np.array([1.0, 1.0])
        old_position = np.array([self.state_history[-1]['lat'], self.state_history[-1]['lon']])
        position = np.array([self.state['lat'], self.state['lon']])
        isBetter = np.linalg.norm(goal - position) < np.linalg.norm(goal - old_position)
        reward = 1 if isBetter else -1

        # Distance from goal
        distance = np.linalg.norm(goal - position)

        # Log the action, new state, and reward
        self.log += f'Action: {self.actions[action].__name__}\n'
        self.log += f'New State: {self.state}\n'
        self.log += f'Reward: {reward}\n'

        # Copy the state to new object
        self.state_history.append(self.state.copy())

        # Update the number of steps left
        self.steps_left -= 1

        # Check if the episode is over
        done = False
        if self.steps_left == 0:
            done = True
        elif distance < 0.01:
            done = True
            reward = 100
        elif position[0] < 0 or position[0] > 1 or position[1] < 0 or position[1] > 1:
            done = True
            reward = -100

        # Return
        return self.observation(), reward, done, {}

    def observation(self):
        return np.array([self.state[key] for key in self.observations])

    def reset(self):
        self.state = {
            'lat': 0,
            'lon': 0,
            'speed': 0.1,
            'heading_lat': 0.1,
            'heading_lon': 0
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
        plt.scatter(path[:,0], path[:,1], c=colors, cmap='cool')
        plt.xlim = (0, 1)
        plt.ylim = (0, 1)
        plt.grid()
        plt.show()

    def close(self):
        pass

def one_step(env):
    # Select an action
    action = env.action_space.sample()

    # Take the action
    observation, reward, done, info = env.step(action)

if __name__ == "__main__":
    print("Hello")

    # Create the environment
    env = CyclingEnv()

    # # Reset the environment
    # observation = env.reset()

    # # Render initial environment
    # env.render()

    # # Run the environment
    # steps = 10

    # for step in range(steps):
    #     one_step(env)

    # # Render final environment
    # env.render(mode='path')

    # Model
    model = PPO('MlpPolicy', env, verbose=1, gamma=0.9)

    # Train the model
    model.learn(total_timesteps=50000)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Render one episode of the model
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            break
    
    env.render(mode='path')
