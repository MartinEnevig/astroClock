import time
from matplotlib import pyplot as plt

fig = plt.figure()
plt.ion()
ax = fig.add_subplot(111)
earth, = ax.plot(0.0, 0.6, 'bo')
venus, = ax.plot(5, 5, 'go')
plt.show()
for i in range(5):
    earth.set_xdata(i)
    earth.set_ydata(5-i)

    venus.set_xdata(5-i)
    venus.set_xdata(5-i)

    fig.canvas.draw()

    

    
    fig.canvas.flush_events()
    time.sleep(0.5)

# from math import pi
# import matplotlib.pyplot as plt
# import numpy as np
# import time
 
# # generating random data values
# x = np.linspace(1, 1000, 5000)
# y = np.random.randint(1, 1000, 5000)
 
# # enable interactive mode
# plt.ion()
 
# # creating subplot and figure
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y)
 
# # setting labels
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Updating plot...")
 
# # looping
# for _ in range(50):
   
#     # updating the value of x and y
#     line1.set_xdata(x*_)
#     line1.set_ydata(y)
 
#     # re-drawing the figure
#     fig.canvas.draw()
     
#     # to flush the GUI events
#     fig.canvas.flush_events()
#     time.sleep(0.1)
