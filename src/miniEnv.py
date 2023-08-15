from gymnasium.spaces.utils import flatten
from gymnasium import Env
from gymnasium.spaces import Box

from matplotlib import pyplot as plt

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import random
import const as c

import numpy as np
import os

from typing import List, Optional, Tuple, Any

class spaceEnvMini(Env):
    """
    Observation space is a box of shape (4,).
    Columns represent:
    0: Current position r
    1: Current position theta
    2: Target position r
    3: Target Position theta

    Action space is a box of shape (2, ) with values between -1 and 1   

    """
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(self, render_mode: Optional[str]=None) -> None:
        # Load planet data from files
        with open('../src/data/data1.npy', 'rb') as file:
            self.position_data_polar = np.load(file)[0:30, :, :]   # Numpy array of body positions in polar coordinates. r,v (5206896, 10, 2)
        #     self.position_data_XY = np.load(file)      # Numpy array of body positions in cortesian coordinates. x,y (5206896, 10, 2)
        #     self.time_data = np.load(file)            # Numpy array of time stamp. year, month, day, hour, minute (5206896, 5)
        # with open('../src/data/data2.npy', 'rb') as file:
        #     self.start_positions = np.load(file)      # Numpy array of allowed start positions = no initial collision (12511,)
        #     self.min_dist_squared = np.load(file)      # Numpy array of 'minimum distance squared' between planets (10, 10)
        
        self.current_step = 0
        self.state: np.ndarray = self.start_state()
    
        # Define action space
        self.action_space = Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]), 
        )
        self.render_mode = render_mode

        # Define observation space 
        self.observation_space = Box(low=0, high=2830.0, shape=(4,), dtype=np.float32)

    def reset(self, seed: Optional[int]=None, options: Optional[dict[str, Any]]=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.state: np.ndarray = self.start_state()

        info = self.generate_info()
    
        return self.state, info

    def generate_info(self, new_position: Optional[Tuple[np.float32, np.float32]]=None) -> dict:
        info = {
            "current round": self.current_step,
            "current position": [self.state[0], self.state[1]],
            "target position": [self.state[2], self.state[3]],
            "new_position": new_position,
            "distance_to_target": (self.polar_to_cartesian(dist=new_position[0], theta=new_position[1]) if new_position else None)
        }
        return info

    def step(self, action: np.ndarray):

        new_position = self.calc_new_position(
            action=action, 
            old_position_dist=self.state[0], 
            old_position_theta=self.state[1]
            )

        distance_to_target = self.cartesian_distance(
                            position_dist=new_position[0],
                            position_theta=new_position[1],
                            target_dist=self.state[2],
                            target_theta=self.state[3]
                        )
        
        reward, truncated = self.calc_reward(distance_to_target=distance_to_target, polar_dist=new_position[0])
        
        info = self.generate_info(new_position=new_position)

        self.current_step+=1

        terminated = (True if self.current_step > 25 else False)

        self.update_state(new_position=new_position)

        return self.state, reward, terminated, truncated, info

    def start_state(self) -> np.ndarray:
        state = np.array([
            self.position_data_polar[self.current_step][0][0],
            self.position_data_polar[self.current_step][0][1],
            self.position_data_polar[self.current_step + 1][0][0],
            self.position_data_polar[self.current_step + 1][0][1],
            ])
        
        return state
    
    def update_state(self, new_position: Tuple[np.float32, np.float32]):
        if new_position[0] > c.BOARD_RADIUS:
            new_dist = c.BOARD_RADIUS
        elif new_position[0] < 0:
            new_dist = 0
        else:
            new_dist = new_position[0]

        new_theta = (new_position[1] - 2*np.pi if new_position[1] > 2*np.pi else new_position[1])

        self.state[0] = new_dist
        self.state[1] = new_theta
        self.state[2] = self.position_data_polar[self.current_step + 1][0][0]
        self.state[3] = self.position_data_polar[self.current_step + 1][0][1]

    def calc_new_position(
            self, 
            action: np.ndarray, 
            old_position_dist: np.float32,
            old_position_theta: np.float32
            ) -> Tuple[np.float32, np.float32]:
        """
        Calculate next position based on acion taken.
        Position is calculated by maping the elements in the action tuple to respectively a [-250, 250] interval,
        and a [0, pi/10] interval, which represents the maximum distance the objects can be moved on the r and theta 
        parameters each turn.
        """
        new_dist = np.float32(old_position_dist + action[0]*250)
        new_theta = np.float32(old_position_theta + ((action[1]+1)/2)*0.1*np.pi) # map [-1, 1] interval to [0, 1] interval, then reduce the step size by multiplying by 0.1 and map it to interval [0, pi]. The model is allowed to moved between 0 and pi/10 degrees.
        if new_theta > 2*np.pi:
            new_theta = np.float32(new_theta - 2*np.pi)

        return (new_dist, new_theta)
    
    
    def cartesian_distance(self,
                         position_dist: np.float32,
                         position_theta: np.float32, 
                         target_dist: np.float32,
                         target_theta: np.float32
                         ) -> np.float32:
        
        position = self.polar_to_cartesian(
                                dist=position_dist,
                                theta=position_theta
                            )
        target = self.polar_to_cartesian(
                                dist=target_dist,
                                theta=target_theta
                            )

        distance = np.float32(abs(np.linalg.norm(position - target)))

        return distance
    
    def calc_reward(self, distance_to_target, polar_dist):
        if polar_dist > c.BOARD_RADIUS or polar_dist < 0:
            reward = -500
            truncated = True
        
        elif distance_to_target < 0.5:
            reward = 10
            truncated = False
        else:
            reward = -(np.abs(distance_to_target))
            truncated = False

        return reward, truncated
    
    @staticmethod
    def polar_to_cartesian(dist: np.float32, theta: np.float32) -> np.ndarray:
        x = np.float32(dist*np.sin(theta))
        y = np.float32(dist*np.cos(theta))

        return np.array([x, y])