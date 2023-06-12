from gym import Env
from gym.spaces import Box
#from viz import StarViz
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import numpy as np
import const as c
from typing import List, Optional

class spaceEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(self, render_mode: Optional[str]=None) -> None:
        # Load planet data from files
        with open('data/data1.npy', 'rb') as file:
            self.PositionDataPolar = np.load(file)   # Numpy array of body positions in polar coordinates. r,v (5206896, 10, 2)
            self.PositionDataXY = np.load(file)      # Numpy array of body positions in cortesian coordinates. x,y (5206896, 10, 2)
            self.TimeData = np.load(file)            # Numpy array of time stamp. year, month, day, hour, minute (5206896, 5)
        with open('data/data2.npy', 'rb') as file:
            self.StartPositions = np.load(file)      # Numpy array of allowed start positions = no initial collision (12511,)
            self.MinDistSquared = np.load(file)      # Numpy array of 'minimum distance squared' between planets (10, 10)
        # Define action space
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        # Define observation space
        # Current position [x,y,r,v] , Target position [x,y,r,v]
        self.observation_space = Box(
            low=np.array(2*[-c.BOARD_RADIUS, -c.BOARD_RADIUS, 0.0, 0.0 ]),
            high=np.array(2*[c.BOARD_RADIUS, c.BOARD_RADIUS, c.BOARD_RADIUS, 2*np.pi]), 
            dtype=np.float32
        )

        self.render_mode = render_mode

    def reset(self):
        # Reset step counter
        self.current_step = 0
        # Select random time index without collition
        self.current_time_index = np.random.choice(self.StartPositions)
        # Set current positions [x,y,r,v]
        self.current_positions = np.concatenate(
            (
            self.PositionDataXY[self.current_time_index],
            self.PositionDataPolar[self.current_time_index]
            ), 
            axis=1
            ) 
        # Next time index
        self.current_time_index += 1
        # Set target positions [x,y,r,v]
        self.target_positions = np.concatenate(
            (
            self.PositionDataXY[self.current_time_index],
            self.PositionDataPolar[self.current_time_index]
            ), 
            axis=1
            ) 
        # Select current planet
        self.it_planets = self.calc_planet_order(self.current_positions)
        self.current_planet = next(self.it_planets)
        # Update state
        self.state = np.concatenate(
            (
            self.current_positions[self.current_planet],
            self.target_positions[self.current_planet]
            )
        )
        return self.state  
    
    def step(self, action: np.ndarray):
        self.current_step += 1
        # Update current position of selected planet 
        r = self.current_positions[self.current_planet, 2]
        v = self.current_positions[self.current_planet, 3] 
        r += action[0] * 100.0              # +/- 100 mm
        v += (action[1] + 1.0) * 0.05       # + 0.1 rad
        if v >= 2.0*np.pi:
            v -= 2.0*np.pi
        x = r * np.sin(v)
        y = r * np.cos(v)
        self.current_positions[self.current_planet] = [x,y,r,v]
        # calculate dist-squared to target
        x_target = self.target_positions[self.current_planet, 0]
        y_target = self.target_positions[self.current_planet, 1]
        distSq = (x-x_target)**2 + (y-y_target)**2
        # Next time index
        self.current_time_index += 1
        isOutOfTime = bool(self.current_time_index >= len(self.PositionDataXY)-2)
            
        # Set new target positions [x,y,r,v]
        self.target_positions = np.concatenate(
            (
            self.PositionDataXY[self.current_time_index],
            self.PositionDataPolar[self.current_time_index]
            ), 
            axis=1
            ) 
        # Calculate reward
        reward = -distSq
        # Done?
        isTooFarFromTarget = bool(distSq >= 90000)
        isOutsideBoard = bool(r > c.BOARD_RADIUS)
        done = isTooFarFromTarget or isOutsideBoard or isOutOfTime
        # Update state
        self.state = np.concatenate(
            (
            self.current_positions[self.current_planet],
            self.target_positions[self.current_planet]
            )
        )

        info = {}
        return self.state, reward, done, info


    def calc_planet_order(self, polarPositions):
        return iter(np.argsort(polarPositions[:,3]))


    
    # def calcDistSquared(posXY1, posXY2):
    #     return (posXY1[0]-posXY2[0])**2 + (posXY1[1]-posXY2[1])**2

    # def calcDist(posXY1, posXY2):
    #     return np.sqrt(calcDistSquared(posXY1, posXY2))

    # def collisionCheck(posXY1, posXY2, minDistSquared):
    #     return (calcDistSquared(posXY1, posXY2) < minDistSquared)

    # def collisionCheckAll(bdyList, MinDistSquared):
    #     for i in range(len(bdyList)-1):
    #         for j in range(i+1, len(bdyList)):
    #             if collisionCheck(bdyList[i], bdyList[j], MinDistSquared[i,j]):
    #                 return True
    #     return False
    

# myEnv = spaceEnv()
# check_env(myEnv)
# print('done')