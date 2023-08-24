import numpy as np
from typing import Union

with open('../src/data/data1.npy', 'rb') as file:
    position_data_polar = np.load(file)[0:30, :, :]

class Celestial:
    def __init__(self,
                 body_id: int,    
                 start_r: np.float32=np.float32(0), 
                 start_theta: np.float32=np.float32(0),
                 ):
        self.r = start_r
        self.theta = start_theta
        self.body_id = body_id
        self.bodies = self.body_list()
        self.name = self.bodies[body_id]["name"]

    def update_position(self, 
                        new_r: Union[float, np.float32], 
                        new_theta: Union[float, np.float32]
                        ):
        """
        Updates an instance's position. Not strictly necessary, but is included for readablity.
        Also, it allows users to input both floats and np.float32.
        """
        self.r = np.float32(new_r)
        self.theta = np.float32(new_theta)

    @staticmethod
    def body_list():
        return {
            0: {"name": "Sun", "radius": 135/2},
            1: {"name": "Jupiter", "radius": 110/2},
            2: {"name": "Neptune", "radius": 110/2},  
            3: {"name": "Saturn", "radius": 110/2},  
            4: {"name": "Uranus", "radius": 110/2},  
            5: {"name": "Moon", "radius": 135/2},  
            6: {"name": "Mercury", "radius": 80/2},  
            7: {"name": "Venus", "radius": 80/2},  
            8: {"name": "Mars", "radius": 80/2},  
            9: {"name": "GalacticCenter", "radius": 80/2},    
        }

#Utility functions

def calculate_collision(next_r: np.float32, 
                        next_theta: np.float32, 
                        nearest_four_ids: list[int]
                        ) -> bool:
    """
    Takes a new position (r, and theta) and a list of id's of the nearest four bodies.
    From this it calculates whether a move to the planned position will result in a collision.
    Returns a bool indcating collision.  
    """
    raise NotImplemented

def moving_order():
    """
    Calculates the order the bodies should be moved in.
    """
    pass

def rotate(step: int, current_body_id: int, nearest_four_ids: list[int]) -> tuple:
    """
    Rotates the board so that the current object to be moved is placed at theta 0. 
    The nearest four are also moved accordingly, so the distance and positions are maintained.
    This avoids the problem of passing from theta 2pi across theta 0. 

    Returns adjusted thetas for current object's target and each of the nearest four.
    """
    current_body_theta = position_data_polar[step][current_body_id][1]

    new_target = passing_zero(
        current_theta=current_body_theta, 
        target_theta=position_data_polar[step+1][current_body_id][1]
    )
    new_theta_neighbour_0 = passing_zero(
        current_theta=current_body_theta,
        target_theta=position_data_polar[step+1][nearest_four_ids[0]][1]
    )
    new_theta_neighbour_1 = passing_zero(
        current_theta=current_body_theta,
        target_theta=position_data_polar[step+1][nearest_four_ids[1]][1]
    )  
    new_theta_neighbour_2 = passing_zero(
        current_theta=current_body_theta,
        target_theta=position_data_polar[step+1][nearest_four_ids[2]][1]
    )    
    new_theta_neighbour_3 = passing_zero(
        current_theta=current_body_theta,
        target_theta=position_data_polar[step+1][nearest_four_ids[3]][1]
    )  
    new_theta_neighbour_4 = passing_zero(
        current_theta=current_body_theta,
        target_theta=position_data_polar[step+1][nearest_four_ids[4]][1]
    )      

    return (
        new_target, 
        new_theta_neighbour_0, 
        new_theta_neighbour_1, 
        new_theta_neighbour_2, 
        new_theta_neighbour_3, 
        new_theta_neighbour_4
        )            


def passing_zero(current_theta: Union[float, np.float32], 
                 target_theta: Union[float, np.float32]
                 ) -> np.float32:
    return np.float32(
                        (
                            (target_theta - current_theta) 
                            if (target_theta - current_theta) > 0 
                            else (target_theta - current_theta) + np.pi*2
                            )
                        )



