from CooperativeRacing_env import RacingEnv, playNEpisodes
from RacingMaps_env import RacingEnv as RacingEnvMaps

# Standard
import numpy as np

# SB3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

max_steps = 5000

if __name__ == "__main__":
    maps = ["../maps/map_10_30_800_800.pkl", "../maps/map_30_50_800_800.pkl", "../maps/map_50_70_800_800.pkl", "../maps/map_70_90_800_800.pkl"]
    # maps = ["../maps/map_10_30_800_800.pkl"]

    # Define observation and action spaces
    old_env = RacingEnvMaps(maps, max_steps)
    old_env = make_vec_env(lambda: old_env, n_envs=1, seed=np.random.randint(0, 10000))
    old_env = VecFrameStack(old_env, n_stack=3)
    observation_space = old_env.observation_space
    action_space = old_env.action_space

    # Load the pretrained model
    pretrained_model = PPO.load('../eval_models/best_model.zip', env=old_env, custom_objects={'observation_space': observation_space, 'action_space': action_space})
    # pretrained_model = PPO.load('../eval_models/best_coop.zip', env=old_env, custom_objects={'observation_space': observation_space, 'action_space': action_space})

    # Create environment
    env = RacingEnv(maps, max_steps, pretrained_model)
    env = make_vec_env(lambda: env, n_envs=1, seed=np.random.randint(0, 10000))
    env = VecFrameStack(env, n_stack=3)

    # Load agent model
    # model = PPO.load("../models/temp_model")
    # model = PPO.load("../eval_models/best_model.zip")
    # model = PPO.load("../eval_models/actually_the_best.zip", env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    model = PPO.load("../eval_models/best_model.zip", env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})


    # Play episodes
    playNEpisodes(10, env, model, max_steps)