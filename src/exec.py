from env import spaceEnv
from planets import Planet
from viz import StarViz
from trajectories import PlanetInit

import numpy as np

def run_env(episodes: int):
    planetMaker = PlanetInit()
    planets = planetMaker()
    env = spaceEnv(planets=planets, render_mode=None)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            env.render()
            n_state, reward, done, info = env.step([0, 0])
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

run_env(1)
