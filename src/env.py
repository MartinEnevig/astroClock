import gym
from gym import Env
from gym.spaces import MultiDiscrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import numpy as np
import os
from typing import List
from planets import Planet


class spaceEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self, planets: List[Planet]) -> None:
        self.action_space = MultiDiscrete([3, 3])
        self.observation_space = Box(low=-100.0, high=100.0, shape=(2, 4), dtype=np.float32)
        #set starting state
        self.state: np.ndarray = np.array(
                        [
                    [-15.0, 0.0, 15.0, 0.0],
                    [-30.0, 0.0 , 15.0, 0.0]          
                ], dtype=np.float32
            )
        self.planets = planets
        self.current_step = 0
        self.window = None
        self.clock = None
    
    def step(self, action):
        self.state = self.calculate_state(action)
        obs: np.ndarray = self.state
        reward = self.calculate_reward()
        
        done = self.is_done()

        info = {
            "venus": self.planets[0].position,
            "earth": self.planets[1].position
        }

        self.current_step+=1
        return obs, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.state: np.ndarray = np.array(
                        [
                    [-15.0, 0.0, 15.0, 0.0],
                    [-30.0, 0.0 , 15.0, 0.0]          
                ],
                dtype=np.float32
            )
        self.current_step = 0
        return self.state

    def calculate_reward(self):
        reward = 0
        for planet in self.planets:
            if np.abs(planet.distance_to_nearest_planet) > 5:
                reward+=0
            else:
                reward-=planet.distance_to_nearest_planet
            if np.abs(planet.distance_to_ideal_position) > 0:
                reward-=np.abs(planet.distance_to_ideal_position)
            if np.array_equal(planet.position, planet.ideal_postion):
                reward+=1
            if np.abs(planet.distance_to_nearest_planet) < 1:
                reward-=100
                
        return reward

    def calculate_state(self, action) -> np.array:
        positions = []
        for action, planet in enumerate(self.planets):
            step = planet.current_step + 1
            x = planet.trajectory[step][0]
            y = planet.get_next_position(action)[1]
            planet.update_position(action)
            planet.current_step = step
            positions.append([x, y])

        distances = []
        for planet in self.planets:
            distance_to_nearest_planet: np.float32 = min(planet.get_distance_to_other_planets().values())      
            planet.distance_to_nearest_planet = distance_to_nearest_planet
            distance_to_ideal_position = planet.get_distance_to_ideal_position()
            distances.append([
                distance_to_nearest_planet,
                distance_to_ideal_position
            ])
        
        state_list = []
        for i in range(len(self.planets)):
            row = [positions[i][0], positions[i][1], distances[i][0], distances[i][1]]
            state_list.append(row)
        
        return np.array(state_list, dtype=np.float32)
    
    def is_done(self):
        if self.current_step==119:
            done = True
        else:
            done = False
        return done
        
    