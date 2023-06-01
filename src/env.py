from gym import Env
from gym.spaces import MultiDiscrete, Box
from viz import StarViz
from planets import Planet

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import numpy as np
from typing import List

class spaceEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(self, planets: List[Planet], render_mode: str) -> None:
        self.action_space = MultiDiscrete([3, 3])
        self.observation_space = Box(low=-100.0, high=100.0, shape=(2, 4), dtype=np.float32)
        self.render_mode = render_mode
        #set starting state
        self.state: np.ndarray = np.array(
                        [
                    [0.0, 4.8, 4.1, 0.0],
                    [0.0, 8.9 , 4.1, 0.0]          
                ], dtype=np.float32
            )
        self.planets = planets
        self.current_step = 0
        self.visualizer = StarViz(planets=self.planets)
        self.clock = None
    
    
    def step(self, action: np.ndarray):
        self.state = self.calculate_state(action)
        obs: np.ndarray = self.state
        reward = self.calculate_reward()
        self.current_step+=1
        self.update_planet_steps()
        done = self.is_done()
        info = {
            "venus": {
                "name": self.planets[0].name,
                "position": self.planets[0].position,
                "distance_to_nearest": self.planets[0].distance_to_nearest_planet,
                "deviation": self.planets[0].deviation,
                "current_step": self.planets[0].current_step
            },
    
            "earth": {
                "name": self.planets[1].name,
                "position": self.planets[1].position,
                "distance_to_nearest": self.planets[1].distance_to_nearest_planet,
                "deviation": self.planets[1].deviation,
                "current_step": self.planets[1].current_step
            }, 
            "step no": self.current_step
        }

        return obs, reward, done, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        self.visualizer.update_plot(planetas=self.planets)
    
    def reset(self):
        self.state: np.ndarray = np.array(
                        [
                    [0.0, 4.8, 4.1, 0.0],
                    [0.0, 8.9 , 4.1, 0.0]            
                ],
                dtype=np.float32
            )
        self.current_step = 0
        for planet in self.planets:
            planet.position = planet.starting_position
            planet.deviation = 0.0
            planet.current_step = 0
        return self.state

    def calculate_reward(self):
        reward = 0
        for planet in self.planets:
            if np.abs(planet.distance_to_nearest_planet) > 5:
                reward+=0
                #print("distance_greater than five")
            else:
                reward-=planet.distance_to_nearest_planet
                #print("distance less than five")
                #print(f"distanceto_idealposition for {planet} is {planet.distance_to_ideal_position}")
            if np.abs(planet.distance_to_ideal_position) > 0:
                reward-=np.abs(planet.distance_to_ideal_position)
                #print("distance to ideal less than five")
            else:
                reward+=1
            if np.abs(planet.distance_to_nearest_planet) < 0.1:
                print("collision")
                reward-=100
                
        return reward

    def calculate_state(self, action: np.ndarray) -> np.ndarray:
        positions = []
        for action_number, planet in enumerate(self.planets):
            planet.update_position(action[action_number])
            positions.append(planet.position)
            planet.update_current_step()

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
    
    def update_planet_steps(self):
        for planet in self.planets:
            if planet.current_step < len(planet.trajectory) - 1:
                planet.current_step = self.current_step
    
    def is_done(self):
        if self.current_step==127:
            done = True
        else:
            done = False
        return done
        
    