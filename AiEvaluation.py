import importlib
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import constants as c
importlib.reload(c)

# Load data from files
with open('AI_data.npy', 'rb') as file:
    PositionDataPolar = np.load(file)   # Numpy array of body positions in polar coordinates. r,v (5206896, 11, 2)
    PositionDataXY = np.load(file)      # Numpy array of body positions in cortesian coordinates. x,y (5206896, 11, 2)
    TimeData = np.load(file)            # Numpy array of time stamp. year, month, day, hour, minute (5206896, 5)
with open('AI_data2.npy', 'rb') as file:
    StartPositions = np.load(file)      # Numpy array of allowed start positions = no initial collision (12511,)
    MinDistSquared = np.load(file)      # Numpy array of 'minimum distance squared' between bodies (11, 11)

class SkyPlot():
    def __init__(self):

        plt.ion()
        self.fig = plt.figure(facecolor='k')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-c.BOARD_RADIUS, c.BOARD_RADIUS)
        self.ax.set_ylim(-c.BOARD_RADIUS, c.BOARD_RADIUS)
        self.ax.set_axis_off()
        self.ax.add_artist(plt.Circle((0, 0), c.BOARD_RADIUS, color='w', fill=False))
        self.ax.add_artist(plt.Circle((0, 0), c.HORIZON_RADIUS, color='w', fill=False))
        self.dateText = self.ax.text(700, 1400, ' ', fontsize=12, color='w')
        self.bodyCircles = [None]*len(c.Bdy)
        self.bodyText = [None]*len(c.Bdy)
        for i in range(len(self.bodyCircles)):
            self.bodyCircles[i] = plt.Circle((0.0, 0.0), c.Bdy[i]['MarkerRadius'], color='r', fill=False)
            self.ax.add_artist(self.bodyCircles[i])
            self.bodyText[i] = self.ax.text(0.0, 0.0, c.Bdy[i]['Abv'], fontsize=12, color='w', horizontalalignment='center',  verticalalignment='center')

    def update(self, bdyPos, time):
        for i in range(bdyPos.shape[0]):
            self.bodyCircles[i].center = bdyPos[i, 0], bdyPos[i, 1]
            self.bodyText[i].set_position((bdyPos[i, 0], bdyPos[i, 1]))
        self.dateText.set_text(pd.Timestamp(time[0], time[1], time[2], time[3], time[4]).strftime('%d-%m-%Y %H:%M'))
        self.fig.canvas.flush_events()

#print(StartPositions[0:50])
print(len(PositionDataPolar))
plt = SkyPlot()
i = 0
while True:
    plt.update(PositionDataXY[i,:,:], TimeData[i,:])
    time.sleep(1)
    i += 1
