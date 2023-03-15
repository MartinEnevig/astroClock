from __future__ import annotations
from typing import List, Dict, Type
import math
import numpy as np
 
class Planet:
    def __init__(self, name, position: np.array):
        self.name = name
        self.planet_names = [
            "Mercury",
            "Venus",
            "Earth",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
        ]
        self.position: np.array = position
        self.ideal_postion: np.array = self.get_ideal_position()
        self.next_position: np.array = self.get_next_position()
        self.direction = self.get_direction()
        self.distance_to_other_planets: list = self.get_distance_to_other_planets()
    
    def get_ideal_position(self, date) -> np.array:
        raise NotImplementedError

    def get_direction(self):
        raise NotImplementedError

    def get_distance_to_other_planets(self, planets: List[Type[Planet]]) -> Dict:
        dist_dict = {}
        for planet in planets:
            if planet.name != self.name:
                distance = math.dist(self.position, planet.position)
                dist_dict[planet.name] = distance
        return dist_dict

    def get_next_position(self, date) -> np.array:
        raise NotImplementedError