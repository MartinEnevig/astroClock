import os
from stable_baselines3 import PPO
from env import spaceEnv
from planets import Planet
from utils.trajectories import PlanetInit
from stable_baselines3.common.evaluation import evaluate_policy

planetmaker = PlanetInit()
planets = planetmaker()
env =spaceEnv(planets=planets, render_mode=None)

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=1000000)


model.save('PPO')

evaluate_policy(model, env, n_eval_episodes=10, render=False)
