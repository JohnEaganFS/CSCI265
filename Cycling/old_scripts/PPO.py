# Notes: They said it couldn't be done. They were right.
# Advantage estimates too large, causing policy to collapse (GAE?)
# Need normalization of observations
# Need support for multi-discrete action spaces for testing, Box for training on image
# Weight initialization seems to matter a lot (paper suggests orthogonal initialization)
# Pytorch ADAM hyperparameters are finicky for some reason
# Mini-batches would probably help (maybe 4096 batch with 128-sized minibatches?)

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.0001
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epochs = 4
T_horizon = 128
batch_size = 4096
num_epochs = 10

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        
        # Define the policy network
        self.policy_layer1 = nn.Linear(state_dim, n_latent_var)
        self.policy_layer2 = nn.Linear(n_latent_var, n_latent_var)
        self.policy_layer3 = nn.Linear(n_latent_var, action_dim)
        
        # Define the value network
        self.value_layer1 = nn.Linear(state_dim, n_latent_var)
        self.value_layer2 = nn.Linear(n_latent_var, n_latent_var)
        self.value_layer3 = nn.Linear(n_latent_var, 1)
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        # Convert state to tensor
        state = torch.from_numpy(state).float()
        
        # Pass state through policy network
        policy = F.relu(self.policy_layer1(state))
        policy = F.relu(self.policy_layer2(policy))
        policy = self.policy_layer3(policy)
        
        # Pass state through value network
        value = F.relu(self.value_layer1(state))
        value = F.relu(self.value_layer2(value))
        value = self.value_layer3(value)
        
        # Convert policy and value to numpy arrays
        policy = policy.detach().numpy()
        value = value.detach().numpy()
        
        # Sample action from policy
        action = np.random.choice(policy.shape[0], p=policy)
        
        return action, value
        
    def evaluate(self, state, action):
        # Convert state to tensor
        state = torch.from_numpy(state).float()
        
        # Pass state through policy network
        policy = F.relu(self.policy_layer1(state))
        policy = F.relu(self.policy_layer2(policy))
        policy = self.policy_layer3(policy)
        
        # Pass state through value network
        value = F.relu(self.value_layer1(state))
        value = F.relu(self.value_layer2(value))
        value = self.value_layer3(value)
        
        # Convert policy and value to numpy arrays
        policy = policy.detach().numpy()
        value = value.detach().numpy()
        
        # Get log probability of action
        action_logprob = torch.log(policy[action])
        
        return action_logprob, value
        
    def get_value(self, state):
        # Convert state to tensor
        state = torch.from_numpy(state).float()
    
        # Pass state through value network
        value = F.relu(self.value_layer1(state))
        value = F.relu(self.value_layer2(value))
        value = self.value_layer3(value)

        # Convert value to numpy array
        value = value.detach().numpy()

        return value
    
# Define the PPO algorithm
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var):
        self.lr = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Initialize the Actor-Critic network
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)
        
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Initialize the memory
        self.memory = []
        
    def act(self, state):
        return self.policy.act(state)
        
    def update(self):
        # Convert memory to numpy array
        memory = np.array(self.memory)
        
        # Get states, actions, rewards, and next states from memory
        states = np.vstack(memory[:, 0])
        actions = np.array(memory[:, 1], dtype=np.int32)
        rewards = np.array(memory[:, 2], dtype=np.float32)
        next_states = np.vstack(memory[:, 3])
        dones = np.array(memory[:, 4], dtype=np.bool)
        
        # Get log probabilities and values for each state-action pair
        logprobs, values = self.policy.evaluate(states, actions)
        
        # Compute advantages
        advantages = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards) - 1):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.lmbda * advantages[t + 1]
        advantages[-1] = rewards[-1] + self.gamma * next_value - values[-1]
        
        # Compute returns
        returns = advantages + values
        
        # Convert everything to tensors
        logprobs = torch.tensor(logprobs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Get new log probabilities and values
            new_logprobs, new_values = self.policy.evaluate(states, actions)
            
            # Compute ratios
            ratios = torch.exp(new_logprobs - logprobs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = F.mse_loss(returns, new_values)

            # Compute total loss
            # Assuming vf_coef = 0.5 for now
            loss = actor_loss + 0.5 * critic_loss

            # TODO: Entropy loss (why is this not working)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Clear memory
        self.memory = []

if __name__ == "__main__":
    # Testing out the PPO algorithm on the LunarLander-v2 environment
    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_latent_var = 64

    # Initialize the PPO algorithm
    ppo = PPO(state_dim, action_dim, n_latent_var)

    # Initialize lists for plotting
    running_reward = 0
    avg_length = 0
    timestep = 0
    timestep_list = []
    reward_list = []
    avg_length_list = []

    # Run the main loop
    for i in range(num_epochs):
        # Reset the environment
        state = env.reset()
        done = False

        # Run the episode
        while not done:
            # Get action
            action, value = ppo.act(state)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Store data in memory
            ppo.memory.append([state, action, reward, next_state, done])

            # Update state
            state = next_state

            # Update lists
            running_reward += reward
            timestep += 1
            avg_length += 1

            # Update the policy
            if timestep % T_horizon == 0:
                ppo.update()
                timestep_list.append(timestep)
                reward_list.append(running_reward / T_horizon)
                avg_length_list.append(avg_length / T_horizon)
                running_reward = 0
                avg_length = 0

        # Print results
        print(f"Episode {i + 1} | Timesteps: {timestep} | Reward: {running_reward}")
    
    # Plot results
    plt.plot(timestep_list, reward_list)
    plt.plot(timestep_list, avg_length_list)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.show()
