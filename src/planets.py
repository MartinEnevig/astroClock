from typing import List, Dict
 
class Planet:
    def __init__(self):
        self.position: list = [0,0]
        self.ideal_postion: list = self.get_ideal_position()
        self.direction = self.get_direction()
        self.distance_to_other_planets: list = self.get_distance_to_other_planets()
    
    def get_ideal_position(self) -> List:
        raise NotImplementedError

    def get_direction(self):
        raise NotImplementedError

    def get_distance_to_other_planets(self, planets: List) -> Dict:
        raise NotImplementedError
