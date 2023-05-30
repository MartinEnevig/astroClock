import matplotlib
from matplotlib import pyplot as plt
from src.planets import Planet
from typing import List, Dict
import time

class StarViz:
    def __init__(self, planets: List[Planet]) -> None:
        self.planets = planets
        self.max_and_min = self.get_max_and_min()
        self.fig = self.get_fig()
        self.sub = self.fig.add_subplot(111)
        self.earthplot = self.make_earthplot()
        self.venusplot = self.make_venusplot()
        plt.ion()

    def get_fig(self):
        fig = plt.figure()
        plt.xlim(self.max_and_min["min_x"], self.get_max_and_min["max_x"])
        plt.ylim(self.max_and_min["min_y"], self.get_max_and_min["max_y"])
        return fig
    
    def get_max_and_min(self) -> Dict[str, int]:
        positions = self.concat_positions()
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for position in positions:
            if position[0] < min_x:
                min_x = position[0]
            elif position[0] > max_x:
                max_x = position[0]
            
            if position[1] < min_y:
                min_y = position[1]
            elif position[1] > max_y:
                max_y = position[1]
        
        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y
        }

    def concat_positions(self) -> List: 
        positions = []
        for planet in self.planets:
            positions = positions + planet.trajectory
        return positions
    
    def make_earthplot(self):
        earthPlot, = self.sub.plot(self.planets[1].position[0], self.planets[1].position[1, 'bo'])
        return earthPlot

    def make_venusplot(self):
        venusPlot, = self.sub.plot(self.planets[0].position[0], self.planets[0].position[1], 'go')
        return venusPlot
    
    def update_plot(self, planetas: List[Planet]):
        self.venusplot.set_xdata(planetas[0].position[0])
        self.venusplot.set_ydata(planetas[0].position[1])
        self.earthplot.set_xdata(planetas[1].position[0])
        self.earthplot.set_ydata(planetas[1].position[1])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)
 