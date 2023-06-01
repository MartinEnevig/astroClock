from typing import List
import sys
from planets import Planet
import numpy as np

class PlanetInit:

    def __init__(self):
        final_trajectory_venus, final_trajectory_earth = self.get_trajectories()

        self.venus = Planet(name="venus", starting_position=np.array([0, 8.9]), trajectory=final_trajectory_venus)
        self.earth = Planet(name="earth", starting_position=np.array([0, 4.8]), trajectory=final_trajectory_earth)

        planets = [self.venus, self.earth]

        self.venus.set_other_planets(planets)
        self.earth.set_other_planets(planets)

    def __call__(self) -> List[Planet]:
        return [self.venus, self.earth]        

    def get_trajectories(self):
        a_1 = 32
        b_1 = 4.8
        trajectory_1 = []
        for i in range(-32,33):
            trajectory_1.append(np.round(self.get_point(x=i, a=a_1, b=b_1), 2))

        trajectory_earth = trajectory_1.copy()
        for el in trajectory_1:
            if el[0]==32 or el[0]==-32:
                continue
            else:
                trajectory_earth.append(np.array([-el[0], -el[1]]))

        a_2 = 16
        b_2 = 8.9
        trajectory_2 = []
        for i in range(-16,17
                    ):
            trajectory_2.append(np.round(self.get_point(x=i, a=a_2, b=b_2), 2))

        trajectory_venus = trajectory_2.copy()
        for el in trajectory_2:
            if el[0]==16 or el[0]==-16:
                continue
            else:
                trajectory_venus.append(np.array([-el[0], -el[1]]))

        #setting starting point to zero
        new_earth_1 = trajectory_earth[32:]
        new_earth_2 = trajectory_earth[0:32]
        final_trajectory_earth = new_earth_1 + new_earth_2

        new_venus_1 = trajectory_venus[16:]
        new_venus_2 = trajectory_venus[0:16]
        final_trajectory_venus = new_venus_1 + new_venus_2 + new_venus_1 + new_venus_2

        return final_trajectory_venus, final_trajectory_earth

    def get_point(self, x: float, a: float, b: float) -> np.array:

        root_number = (1-(x**2/a**2))*b**2
        if root_number < 0:
            root_number*-1
            y = np.sqrt(root_number)
            return np.array([x, -y])
        else:
            y = np.sqrt(root_number)
            return np.array([x, y]) 
        

