from RacingMaps_env import RacingEnv, playNEpisodes

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
    # Create environment
    env = RacingEnv(maps, max_steps)
    env = make_vec_env(lambda: env, n_envs=1, seed=np.random.randint(0, 10000))
    env = VecFrameStack(env, n_stack=3)

    # Load agent model
    # model = PPO.load("../models/temp_model")
    model = PPO.load("../eval_models/best_model.zip")


    # Play episodes
    playNEpisodes(10, env, model, max_steps)