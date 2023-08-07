import os
from env import spaceEnv

from gymnasium.wrappers import flatten_observation

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


env =spaceEnv(render_mode=None)

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)


model.save('PPO')

evaluate_policy(model, env, n_eval_episodes=10, render=False)
