
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import numpy as np
import scipy.integrate as sci


def dX1dt(x, t=0):
    dxdt = np.array([a1 + x[1] * x[0] ** 2 - b1 * x[0] - x[0], b1 * x[0] - x[1] * x[0] ** 2])
    return dxdt
def dX2dt(x, t=0):
    dxdt = np.array([a2 + x[1] * x[0] ** 2 - b2 * x[0] - x[0], b2 * x[0] - x[1] * x[0] ** 2])
    return dxdt

x0 = 0.75
y0 = 2.0
t0 = 0
time_stop = 60
length = 10000
X0 = [x0, y0]

a1 = 1.0
a2 = 1.0
b1 = 1.7
b2 = 2.2


plt.style.use("ggplot")

fig = grid.GridSpec(5,2)
ax1 = plt.subplot(fig[1:3,0])
ax2 = plt.subplot(fig[3:,0])
ax3 = plt.subplot(fig[1:3,1])
ax4 = plt.subplot(fig[3:,1])


time = np.linspace(t0, time_stop, length)
X1 = sci.odeint(dX1dt, X0, time)
x1,y1 = X1.T

ax1.plot(time, x1)
ax1.plot(time, y1)

ax2.plot(x1,y1)

X2 = sci.odeint(dX2dt, X0, time)
x2,y2 = X2.T

ax3.plot(time, x2)
ax3.plot(time, y2)

ax4.plot(x2,y2)

# Now label each plot

ax1.set_title("A", loc="left", fontsize=12)
ax3.set_title("B", loc="right", fontsize=12)

ax1.set_ylabel("Concentration")
ax1.set_xlabel("time")
ax3.set_ylabel("Concentration")
ax3.set_xlabel("time")

ax2.set_ylabel("y")
ax2.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_xlabel("x")

plt.suptitle("Oscillatory Dynamics in the Brusselator", fontsize=16)
plt.tight_layout()
plt.show()