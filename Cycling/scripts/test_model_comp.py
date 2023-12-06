from CompetitiveRacing_env import RacingEnv, playNEpisodes
from RacingMaps_env import RacingEnv as RacingEnvMaps

# Standard
import numpy as np

# SB3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# Single-Agent
# agent_1 = 630000
# filename = "/single_agent/comp_model_" + str(agent_1) + "_steps.zip"
# other_agent_filename = "/single_agent/comp_model_" + str(agent_1) + "_steps.zip" # Doesn't matter unless you're doing multi-agent
# num_agents = 1

# Multi-Agent (No Draft)
# filename = "/multi_no_draft/temp_model copy 3.zip"
# other_agent_filename = "/multi_no_draft/temp_model copy 3.zip"
# num_agents = 2

# Multi-Agent (Draft)
agent_1 = 800000
agent_2 = 800000
filename = "/multi_draft/comp_model_" + str(agent_1) + "_steps.zip"
other_agent_filename = "/multi_draft/comp_model_" + str(agent_2) + "_steps.zip"
num_agents = 2


max_steps = 2000

if __name__ == "__main__":
    # Map used for training (probably overfitted)
    maps = ["../maps/map_70_90_800_800.pkl"]

    # Other maps you can try
    # maps = ["../maps/map_10_30_800_800.pkl"], "../maps/map_30_50_800_800.pkl", "../maps/map_50_70_800_800.pkl", "../maps/map_70_90_800_800.pkl"]

    # Define observation and action spaces
    old_env = RacingEnvMaps(maps, max_steps)
    old_env = make_vec_env(lambda: old_env, n_envs=1, seed=np.random.randint(0, 10000))
    old_env = VecFrameStack(old_env, n_stack=3)
    observation_space = old_env.observation_space
    action_space = old_env.action_space

    # Load the pretrained model
    pretrained_model = PPO.load('../test_models/' + filename, env=old_env, custom_objects={'observation_space': observation_space, 'action_space': action_space})

    # Create environment
    env = RacingEnv(maps, max_steps, num_agents, pretrained_model, evaluating=True, model_filename=other_agent_filename, obs_space_temp=observation_space, act_space_temp=action_space)
    env = make_vec_env(lambda: env, n_envs=1, seed=np.random.randint(0, 10000))
    env = VecFrameStack(env, n_stack=3)

    # Load agent model
    model = PPO.load('../test_models/' + filename, env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})

    # Play episodes
    playNEpisodes(10, env, model, max_steps)