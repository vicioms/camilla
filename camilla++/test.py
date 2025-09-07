import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.linspace(0, 3), np.exp(-3*np.linspace(0, 3)) - np.exp(-np.linspace(0, 3)))
plt.show()

exit()
data = np.loadtxt("test.txt")

fig, ax = plt.subplots()
for t in [-1]:
    ax.plot([data[t,0], data[t,0] + data[t,2]], [data[t,1], data[t,1] + data[t,3]], color='red')
    ax.plot([0, data[t,4]], [0.0, 0.0], color='black')
ax.set_aspect('equal')
plt.show()