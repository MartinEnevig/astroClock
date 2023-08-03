from gymnasium import Env
from gymnasium.spaces import Box

import random
import const as c

import numpy as np
from typing import List, Optional, Any, Tuple

class spaceEnv(Env):
    """
    Observation space is a box of shape (13,).
    Columns represent:
    0: Current position r
    1: Current position theta
    2: Target position r
    3: Target Position theta
    4: Distance to target position
    5: Nearest object r
    6: Nearets object theta
    7: Distance to nearest objects
    8: Second nearest object r
    9: Second nearest object theta
    10: Distance to second nearest object
    11: Third nearest object r
    12. Third nearest object theta
    13: Distance to third nearest object
    14: Fourth nearest object r
    15: fourth nearest object theta
    16: Distance to fourth nearest object
    Rows represent objects. Current object in 0, and the rest in order of closeness.

    Positions are in polar coordinates. Previous and current position will be the same for the objects 
    not being moved this turn.

    Action space is a box of shape (2, ) with values between -1 and 1. It is recommended to normalize action space.
    We don't want the mode to be able to pick negative actions for theta, so for that we map the [-1, 1] interval to
    a [0, 1] interval by adding 1 and dividing by 2.
    """
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(self, render_mode: Optional[str]=None, full_mode: bool = True) -> None:
        # Load planet data from files
        with open('src/data/data1.npy', 'rb') as file:
            self.position_data_polar = np.load(file)   # Numpy array of body positions in polar coordinates. r,v (5206896, 10, 2)
            self.position_data_XY = np.load(file)      # Numpy array of body positions in cortesian coordinates. x,y (5206896, 10, 2)
            self.time_data = np.load(file)            # Numpy array of time stamp. year, month, day, hour, minute (5206896, 5)
        with open('src/data/data2.npy', 'rb') as file:
            self.start_positions = np.load(file)      # Numpy array of allowed start positions = no initial collision (12511,)
            self.min_dist_squared = np.load(file)      # Numpy array of 'minimum distance squared' between planets (10, 10)
        self.full_mode = full_mode
        
        self.objects: List[str] = self.get_object_list()
        self.current_round = self.get_starting_round(full_mode)   
        self.object_positions: dict[str, dict[str, np.float32]] = self.get_starting_positions()
        self.object_count: int = int(0)
        self.current_object: str = self.update_current_object()
        self.state: np.ndarray = self.update_state(initialize=True)
    
        # Define action space
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]), 
            )
        self.render_mode = render_mode

        # Define observation space 
        self.observation_space = Box(low=-c.BOARD_RADIUS*2, high=c.BOARD_RADIUS*2, shape=(17,), dtype=np.float32)
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

        # self.current_object = self.choose_current_object() # Needed for production but currently switched off
        nearest_four = self.calculate_nearest_four()

        current_position_dist = self.object_positions[self.current_object]["position_polar_dist"]
        current_position_theta = self.object_positions[self.current_object]["position_polar_theta"]

        target_position_dist = self.object_positions[self.current_object]["target_position_dist"]
        target_position_theta = self.object_positions[self.current_object]["target_position_theta"]

        new_position = self.calc_new_position(action=action, 
                                              old_position_dist=current_position_dist,
                                              old_position_theta=current_position_theta)
        
        self.state = self.update_state(
            initialize=False,
            nearest_four=nearest_four,
            dist=new_position[0],
            theta=new_position[1]
        )
        
        reward = self.calc_reward()

        terminated = (True if self.current_round==self.position_data_polar.shape[0] and self.object_count == 9 else False)        

        truncated = (True if -c.BOARD_RADIUS < new_position[0] > c.BOARD_RADIUS else False)

        info = {
            "current_body": self.current_object,
            "current_position": (current_position_dist, current_position_theta),
            "new_position": new_position,
            "target_position": (target_position_dist, target_position_theta),
            "reward": reward,
            "object_count": self.object_count,
            "current_step": self.current_round
        }

        self.update_position_dict()

        self.update_current_round()
        self.update_all_positions()

        return self.state, reward, terminated, truncated, info

    def reset(self, seed: Optional[int]=None, options: Optional[dict[str, Any]]=None):
        super().reset(seed=seed, options=options)
        # Reset current_round and object_count
        self.current_round = self.get_starting_round(self.full_mode)
        self.object_count = 0

        # reset objects and pop_list
        self.objects = self.get_object_list()
        # self.pop_list = self.get_object_list() #switched off for training
        
        # reset object_positions
        self.object_positions = self.get_starting_positions()
        
        # reset obs
        self.state = self.update_state(initialize=True)
        
        # info variable to make check_env happy
        info = {}

        return self.state, info
    
    def render(self):
        pass

    def calc_reward(self) -> float:
        """
        Calculate reward.
        The reward for falling off the board is very negative.
        Reward for distance to target quickly gets quite hihg as it is squared. Therefore it is divided by 100.
        The logic behind squaring it is, that th epunishment should grow the further the body is from its 
        target position.

        The reward for being on target - defined as a euclidian distance of less than 0.5, which is just a numebr
        i came up with is set to a 1000 to try to balance the negative rewards. Probably needs to be fine-tuned.
        """

        reward = 0.0

        distance_to_target = self.state[4]

        if -c.BOARD_RADIUS < self.state[0] > c.BOARD_RADIUS:
            reward = -5_000_000.0
        
        elif distance_to_target < 0.5:
            reward = 1_000.0
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
   
    def update_state(self, 
                     initialize: bool,
                     nearest_four: Optional[List] = None,
                     dist: np.float32 = np.float32(0.0),
                     theta: np.float32 = np.float32(0.0),
                     ) -> np.ndarray:
        # initializing a matrix of zeros in float32 to hold observations
        if initialize:
            state = np.zeros(shape=(17,), dtype=np.float32)
            position_dist = self.object_positions[self.current_object]["position_polar_dist"]
            position_theta = self.object_positions[self.current_object]["position_polar_theta"]
            nearest_objects = self.calculate_nearest_four()
        else:
            state = self.state
            position_dist = dist
            position_theta = theta
            nearest_objects = nearest_four
        
        target_position_dist = self.object_positions[self.current_object]["target_position_dist"]
        target_position_theta = self.object_positions[self.current_object]["target_position_theta"] 
        distance_to_target = self.cartesian_distance(
            current_position_dist=position_dist,
            current_position_theta=position_theta,
            new_position_dist=target_position_dist,
            new_position_theta=target_position_theta
        )
        
        nearest_objects = self.calculate_nearest_four()
        nearest_dist = self.object_positions[nearest_objects[0]]["position_polar_dist"]
        nearest_theta = self.object_positions[nearest_objects[0]]["position_polar_theta"]
        second_nearest_dist = self.object_positions[nearest_objects[1]]["position_polar_dist"]
        second_nearest_theta = self.object_positions[nearest_objects[1]]["position_polar_theta"]
        third_nearest_dist = self.object_positions[nearest_objects[2]]["position_polar_dist"]
        third_nearest_theta = self.object_positions[nearest_objects[2]]["position_polar_theta"]
        fourth_nearest_dist = self.object_positions[nearest_objects[3]]["position_polar_dist"]
        fourth_nearest_theta = self.object_positions[nearest_objects[3]]["position_polar_theta"]

        state[0] = position_dist
        state[1] = position_theta
        state[2] = target_position_dist
        state[3] = target_position_theta
        state[4] = distance_to_target
        state[5] = nearest_dist
        state[6] = nearest_theta
        nearest_distance = self.cartesian_distance(
            current_position_dist=position_dist,
            current_position_theta=position_theta,
            new_position_dist=nearest_dist,
            new_position_theta=nearest_theta
        )
        state[7] = (1000 if nearest_distance > 1000 else nearest_distance)
        
        state[8] = second_nearest_dist
        state[9] = second_nearest_theta
        second_nearest_distance = self.cartesian_distance(
            current_position_dist=position_dist,
            current_position_theta=position_theta,
            new_position_dist=second_nearest_dist,
            new_position_theta=second_nearest_theta
        )
        state[10] = (1000 if second_nearest_distance > 1000 else second_nearest_distance)
        
        
        state[11] = third_nearest_dist
        state[12] = third_nearest_theta
        third_nearest_distance = self.cartesian_distance(
            current_position_dist=position_dist,
            current_position_theta=position_theta,
            new_position_dist=third_nearest_dist,
            new_position_theta=third_nearest_theta
        )
        state[13] = (1000 if third_nearest_distance > 1000 else third_nearest_distance)
        
        state[14] = fourth_nearest_dist
        state[15] = fourth_nearest_theta
        fourth_nearest_distance = self.cartesian_distance(
            current_position_dist=position_dist,
            current_position_theta=position_theta,
            new_position_dist=fourth_nearest_dist,
            new_position_theta=fourth_nearest_theta
        )
        state[16] = (1000 if fourth_nearest_distance > 1000 else fourth_nearest_distance)

        return state 
    
    # def choose_current_object(self) -> str:
    #     """
    #     For now this is a dummy method, that just returns a popped object from the object list.
        
    #     When the full implementation is done, this should be a method to find the body that can be most 
    #     easily moved.
    #     """
    #     obj = self.pop_list.pop()
    #     self.object_count+=1
    #     return obj
    
    def calculate_nearest_four(self) -> List[str]:
        """For now this is a dummy method, that returns four random planets from the object list.
        
        When the full implementation is done, this method should identify the four bodies closest to
        the current body.
        """
        nearest = []

        while len(nearest) < 4:
            body = random.sample(self.objects, 1)[0]
            if body not in nearest and body != self.current_object:
                nearest.append(body)
        
        return nearest

    def update_current_round(self):
        """
        Update round.
        This method increments the round by one. If env has reached the final position for the current object,
        object_count is incremented by one, and current round is reset to 0. 
        """
        # self.pop_list = self.get_object_list() #could be necessary in production
        self.current_round+=1
        if self.current_round > self.position_data_polar.shape[0]:
            self.object_count+=1
            self.current_round = 0
            self.current_object = self.update_current_object()

    def update_current_object(self) -> str:
        return self.objects[self.object_count]
    
    def update_position_dict(self):
        """
        Updates the current_object's position in self.object_positions based on the new position
        calculated in step().
        """
        self.object_positions[self.current_object]["position_polar_dist"] = self.state[0]
        self.object_positions[self.current_object]["position_polar_theta"] = self.state[1]
        self.object_positions[self.current_object]["distance_to_target"] = self.state[2]

    def update_all_positions(self):
        """
        Updates the target position for all bodies. 
        To be run when updating round. 
        """
        for entry in enumerate(self.object_positions.keys()):
            self.object_positions[entry[1]]["target_position_dist"] = self.position_data_polar[self.current_round][entry[0]][0]
            self.object_positions[entry[1]]["target_position_theta"] = self.position_data_polar[self.current_round][entry[0]][1]
            if entry[1] != self.current_object:
                self.object_positions[entry[1]]["target_position_dist"] = self.position_data_polar[self.current_round][entry[0]][0]
                self.object_positions[entry[1]]["target_position_theta"] = self.position_data_polar[self.current_round][entry[0]][1]
    
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
    
    @staticmethod
    def polar_to_cartesian(dist: np.float32, theta: np.float32) -> np.ndarray:
        x = np.float32(dist*np.cos(theta))
        y = np.float32(dist*np.sin(theta))

        return np.array([x, y])