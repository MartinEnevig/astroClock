from __future__ import annotations
from typing import List, Dict
import math
import numpy as np
 
class Planet:
    def __init__(self, name: str, position: np.array, trajectory: List):
        self.name: str = name
        self.trajectory: List = trajectory
        self.planets: List[Planet] = []
        self.position: np.ndarray = position
        self.ideal_postion: np.ndarray = None
        self.next_position: np.ndarray = None
        self.distance_to_other_planets: List[np.ndarray] = []
        self.current_step = 0
        self.deviation: np.float32 = 0
        self.distance_to_ideal_position: np.float32 = 0
        self.distance_to_nearest_planet: np.float32 = 0
    
    def set_other_planets(self, other_planets: List) -> List[Planet]:
        self.planets = other_planets
        
    def get_ideal_position(self):
        ideal_position = self.trajectory[self.current_step]
        self.ideal_postion = ideal_position

    def get_direction(self):
        raise NotImplementedError

    def get_distance_to_other_planets(self) -> Dict[str, np.float32]:
        dist_dict = {}
        for planet in self.planets:
            if planet.name != self.name:
                distance = math.dist(self.position, planet.position)
                dist_dict[planet.name] = distance
        return dist_dict

    def get_next_position(self, action) -> np.ndarray:
        deviation = self.get_deviation(action)
        if self.deviation < 10:
            self.deviation += deviation
        next_y: np.float32 = self.trajectory[self.current_step+1][1] + self.deviation
        next_position = np.array([self.current_step+1, next_y], dtype=np.float32)
        return next_position

    def update_current_step(self):
        self.current_step = self.current_step + 1
    
    def get_deviation(self, action: int) -> np.float32:
        if action==1:
            deviation = -0.1
        elif action==2:
            deviation = 0.1
        else:
            deviation = 0
        return deviation

    def get_distance_to_ideal_position(self) -> np.float32:
        self.get_ideal_position()
        distance: np.float32 = math.dist(self.position, self.ideal_postion)
        return distance

    def update_position(self, action):
        self.position = self.get_next_position(action)
    
