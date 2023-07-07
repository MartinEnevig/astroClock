from gymnasium import Env
from gymnasium.spaces import Box

from matplotlib import pyplot as plt


import random
import const as c

import numpy as np


from typing import List, Optional, Any

class spaceEnv(Env):
    """
    Observation space is a box of shape (5, 4).
    Columns represent:
    0: Polar postiion r
    1: Polar position theta,
    2: Distance to target position
    3: Distance to the object currently being moved. For the current object this is set to negative.
    Rows represent objects. Current object in 0, and the rest in order of closeness.

    Action space is a box of shape (2, ) with values between -1 and 1. It is recommended to normalize action space,
    so even though we mever want the model to actually pick a negative value for the theta of the polar coordinates,
    we still allow the possibility.
    """
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(self, render_mode: Optional[str]=None, full_mode: bool = True) -> None:
        # Load planet data from files
        with open('astroClock/src/data/data1.npy', 'rb') as file:
            self.position_data_polar = np.load(file)   # Numpy array of body positions in polar coordinates. r,v (5206896, 10, 2)
            self.position_data_XY = np.load(file)      # Numpy array of body positions in cortesian coordinates. x,y (5206896, 10, 2)
            self.time_data = np.load(file)            # Numpy array of time stamp. year, month, day, hour, minute (5206896, 5)
        with open('astroClock/src/data/data2.npy', 'rb') as file:
            self.start_positions = np.load(file)      # Numpy array of allowed start positions = no initial collision (12511,)
            self.min_dist_squared = np.load(file)      # Numpy array of 'minimum distance squared' between planets (10, 10)
        self.full_mode = full_mode
        
        self.objects: List[str] = self.get_object_list()
        self.current_round = self.get_starting_round(full_mode)   
        self.object_positions: dict[str, dict[str, np.float32]] = self.get_starting_positions()
        self.pop_list = self.get_object_list() # Dummy prop to be replaced in final implementation 
        self.object_count: int = 0
    
        # Define action space
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]), 
            )
        self.render_mode = render_mode

        # Define observation space 
        self.observation_space = Box(low=-2830.0, high=2830.0, shape=(5, 4), dtype=np.float32)
        self.horizon = c.HORIZON_RADIUS      
        
    def step(self, action: np.ndarray):
        """
        Take a step in the model.
        A step is defined as moving one body. Each step picks a body as current_object. Then it selects the 
        nearest for bodies. These are not relevant in the simple implementation, since the only aim here is for the 
        body to be moved to the target location and not fall of the board, but they will be relevant in the next 
        iteration. For now, they are randomly selected.

        The target position for the current object is extracted from the prop self.object_locations.
        A new position is calculated based on the action taken.

        Then the distance to the target is calculated, and the reward, truncated and terminated are calculated
        from this new position. Finally, object_positions is updated with the current_object's new position.

        There is a prop named self.object_count that keeps track of how many objects have been moved in the 
        current round. If object_count is nine all objects have been move. Then self.current_round is 
        incremented by one, and the targets of each body is updated.  
        """

        current_object = self.choose_current_object()
        nearest_four = self.calculate_nearest_four(current_object=current_object)

        target_position_dist = self.object_positions[current_object]["target_position_dist"]
        target_position_theta = self.object_positions[current_object]["target_position_theta"]

        new_position = self.calc_new_position(action=action, current_object=current_object)

        current_position = [
            self.object_positions[current_object]["position_polar_dist"],
            self.object_positions[current_object]["position_polar_theta"]
        ]

        distance_to_target = self.cartesian_distance(
                            current_position_dist=current_position[0],
                            current_position_theta=current_position[1],
                            new_position_dist=target_position_dist,
                            new_position_theta=target_position_theta
                        )
        
        reward = self.calc_reward(distance_to_target=distance_to_target, polar_dist=new_position[0])
        
        obs = self.observation(
            new_position=new_position,
            distance_to_target=distance_to_target,
            nearest_objects=nearest_four
        )

        terminated = (True if self.current_round==self.position_data_polar.shape[0] and self.object_count == 9 else False)        

        truncated = (True if new_position[0] > self.horizon else False)

        info = {
            "current_body": current_object,
            "current_position": current_position,
            "new_position": new_position,
            "target_position": [target_position_dist, target_position_theta],
            "reward": reward,
            "object_count": self.object_count,
            "current_step": self.current_round
        }

        self.update_position(
            current_object=current_object,
            new_position=new_position,
            distance_to_target=distance_to_target
            )

        if self.object_count==9:
            self.update_current_step()
            self.update_targets()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int]=None, options: Optional[dict[str, Any]]=None):
        super().reset(seed=seed, options=options)
        # Reset current_round and object_count
        self.current_round = self.get_starting_round(self.full_mode)
        self.object_count = 0

        # reset objects and pop_list
        self.objects = self.get_object_list()
        self.pop_list = self.get_object_list()
        
        # reset object_positions
        self.object_positions = self.get_starting_positions()
        
        # reset obs
        obs = np.array(self.reset_obs())
        
        # info variable to make check_env happy
        info = {}

        return obs, info
    
    def render(self):
        pass

    def calc_reward(self, distance_to_target, polar_dist):
        """
        Calculate reward.
        The reward for falling off the board is very negative.
        Reward for distance to target quickly gets quite hihg as it is squared. Therefore it is divided by 100.
        The logic behind squaring it is, that th epunishment should grow the further the body is from its 
        target position.

        The reward for being on target - defined as a euclidian distance of less than 0.5, which is just a numebr
        i came up with is set to a 1000 to try to balance the negative rewards. Probably needs to be fine-tuned.
        """
        if polar_dist > self.horizon:
            reward = -5_000_000
        
        elif distance_to_target < 0.5:
            reward = 1_000
        else:
            reward = -(distance_to_target**2)/100

        return reward
    
    def get_object_list(self) -> List[str]:
        """Getting the list of objects from the const file"""
        objects = c.objects.copy()
        return objects

    def get_starting_round(self, mode: bool) -> int:
        """
        Get the starting round.
        If you run the full mode, starting round is chosen randomly from one of the 'safe'
        positions. If not starting round is zero.
        """
        if mode:
            position = int(np.random.choice(self.start_positions))
        else:
            position = 0
        return position  
   
    def choose_current_object(self) -> str:
        """
        For now this is a dummy method, that just returns a popped object from the object list.
        
        When the full implementation is done, this should be a method to find the body that can be most 
        easily moved.
        """
        obj = self.pop_list.pop()
        self.object_count+=1
        return obj
    
    def calculate_nearest_four(self, current_object: str) -> List[str]:
        """For now this is a dummy method, that returns four random planets from the object list.
        
        When the full implementation is done, this method should identify the four bodies closest to
        the current body.
        """
        nearest = []

        while len(nearest) < 4:
            body = random.sample(self.objects, 1)
            if body not in nearest:
                nearest.append(body)
        
        return nearest

    def update_current_step(self):
        """
        Update round.
        This method increments the round by one, restores the pop_list and sets the object count to zero.
        it is to be called when all bodies have been moved.
        """
        self.pop_list = self.get_object_list()
        self.current_round+=1
        self.object_count = 0

    
    def update_position(
            self,
            current_object: str,
            new_position: List[np.float32],
            distance_to_target: np.float32):
        """
        Updates the current_object's position in self.object_positions based on the new position
        calculated in step().
        """
        self.object_positions[current_object]["position_polar_dist"] = new_position[0]
        self.object_positions[current_object]["position_polar_theta"] = new_position[1]
        self.object_positions[current_object]["distance_to_target"] = distance_to_target

    def update_targets(self):
        """
        Updates the target position for all bodies. 
        To be run when updating round. 
        """
        for entry in enumerate(self.object_positions.keys()):
            self.object_positions[entry[1]]["target_position_dist"] = self.position_data_polar[self.current_round][entry[0]][0]
            self.object_positions[entry[1]]["target_position_theta"] = self.position_data_polar[self.current_round][entry[0]][1]

    def calc_new_position(self, action: np.ndarray, current_object: str) -> List[np.float32]:
        new_dist = np.float32(self.object_positions[current_object]["position_polar_dist"] + action[0]*250)
        new_theta = np.float32(self.object_positions[current_object]["position_polar_theta"] + action[1]*2*np.pi)
        return [new_dist, new_theta]
    
  
    def cartesian_distance(
            self,
            current_position_dist: np.float32,
            current_position_theta: np.float32, 
            new_position_dist: np.float32,
            new_position_theta: np.float32
            ) -> np.float32:
        
        current_cartesian = self.polar_to_cartesian(
                                dist=current_position_dist,
                                theta=current_position_theta
                            )
        new_cartesian = self.polar_to_cartesian(
                                dist=new_position_dist,
                                theta=new_position_theta
                            )

        distance = np.float32(abs(np.linalg.norm(current_cartesian - new_cartesian)))

        return distance
    
    @staticmethod
    def polar_to_cartesian(dist: np.float32, theta: np.float32) -> np.ndarray:
        x = np.float32(dist*np.cos(theta))
        y = np.float32(dist*np.sin(theta))

        return np.array([x, y])


    # def calc_planet_order(self, polarPositions):
    #     return iter(np.argsort(polarPositions[:,3]))

    def observation(
            self, 
            new_position: List[np.float32],  
            distance_to_target: np.float32,
            nearest_objects:List[str]
                ) -> np.ndarray:
        """
        Returns observation.
        Observation is position (dist and theta) and target position for current object and four 
        nearest objects.
        """

        # initializing a matrix of zeros in float32 to hold observations
        obs = self.zeros_in_32()

        # populating matrix with current objects updated data
        obs[0, 0] = np.float32(new_position[0])
        obs[0, 1] = np.float32(new_position[1])
        obs[0, 2] = np.float32(distance_to_target)
        obs[0, 3] = np.float32(-99.0)

        # populating matrix with objects that have not been moved
        for object in enumerate(nearest_objects):
            row = object[0] + 1
            dist = np.float32(self.object_positions[object[1][0]]["position_polar_dist"])
            theta = np.float32(self.object_positions[object[1][0]]["position_polar_theta"])

            obs[row, 0] = np.float32(dist)
            obs[row, 1] = np.float32(theta)
            obs[row, 2] = np.float32(self.object_positions[object[1][0]]["distance_to_target"])

            distance_to_current = self.cartesian_distance(
                                        current_position_dist=dist,
                                        current_position_theta=theta,
                                        new_position_dist=np.float32(new_position[0]),
                                        new_position_theta=np.float32(new_position[1])
                                    )
            obs[row, 3] = np.float32(distance_to_current)

        return np.array(obs)

    def reset_obs(self) -> np.ndarray:
        obs = self.zeros_in_32()

        for obj in enumerate(self.objects[0:5]):
            obs[obj[0], 0] = np.float32(self.object_positions[obj[1]]["position_polar_dist"])
            obs[obj[0], 1] = np.float32(self.object_positions[obj[1]]["position_polar_theta"])
            obs[obj[0], 2] = np.float32(0.0)
            obs[obj[0], 3] = np.float32(-99.0)

        return np.array(obs)

    @staticmethod
    def zeros_in_32() -> np.ndarray:
        """
        Initializes a matrix of zeros in float32. 
        
        Cannot use np.zeros as it initializes in float 64 whoch messes up the env.
        """
        return np.array(
            [
                [np.float32(0), np.float32(0), np.float32(0), np.float32(0)], 
                [np.float32(0), np.float32(0), np.float32(0), np.float32(0)],            
                [np.float32(0), np.float32(0), np.float32(0), np.float32(0)],            
                [np.float32(0), np.float32(0), np.float32(0), np.float32(0)],            
                [np.float32(0), np.float32(0), np.float32(0), np.float32(0)],            
                    ]
        )
    def get_starting_positions(self) -> dict[str, dict[str, np.float32]]:
        return {
            "sun":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][0][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][0][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][0][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][0][1]),
                "distance_to_target": np.float32(0)   
            },
            "jupiter":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][1][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][1][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][1][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][1][1]),   
                "distance_to_target": np.float32(0)  
            },
            "neptune":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][2][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][2][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][2][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][2][1]),   
                "distance_to_target": np.float32(0)  
            },
            "saturn":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][3][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][3][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][3][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][3][1]),   
                "distance_to_target": np.float32(0)  
            },
            "uranus":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][4][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][4][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][4][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][4][1]),   
                "distance_to_target": np.float32(0)  
            },
            "moon":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][5][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][5][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][5][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][5][1]),   
                "distance_to_target": np.float32(0)  
            },
            "mercury":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][6][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][6][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][6][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][6][1]),   
                "distance_to_target": np.float32(0)  
            },
           "venus":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][7][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][7][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][7][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][7][1]),   
                "distance_to_target": np.float32(0)  
            },
            "mars":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][8][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][8][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][8][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][8][1]),   
                "distance_to_target": np.float32(0)  
            },
            "galacticcenter":{ 
                "position_polar_dist": np.float32(self.position_data_polar[self.current_round][9][0]),
                "position_polar_theta": np.float32(self.position_data_polar[self.current_round][9][1]),
                "target_position_dist": np.float32(self.position_data_polar[self.current_round+1][9][0]),
                "target_position_theta": np.float32(self.position_data_polar[self.current_round+1][9][1]),   
                "distance_to_target": np.float32(0)  
            }
        }