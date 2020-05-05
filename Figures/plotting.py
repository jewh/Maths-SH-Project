import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import matplotlib.gridspec as gridspec

a = 1.0 # Fix this at 1 for ease
b = 1.9 # Vary this one <2 for damped oscillations, >=2 for sustained
x0 = 0.75
y0 = 2.0 # keeping same ratio of x0:y0 as in Giorgo's code
omega = 300
t0 = 0
time_stop = 60
# specify the name of the file to load matlab data from
matlab_file_name = "lnatrial1.9Omega5000.csv"


# Define the system vector

X0 = np.array([x0, y0])

# define the brusselator

def dXdt(x, a, b, t=0):
    dxdt = np.array([a + x[1]*x[0]**2 - b*x[0] - x[0], b*x[0] - x[1]*x[0]**2])
    return dxdt

# Create code to make side-by-side figures of different qualitative behaviours in the Brusselator





















# Now load in Matlab data
current_wd = os.path.dirname(os.path.realpath(__file__))

# load in as a pandas dataframe
data = pd.read_csv(f"{current_wd}/{matlab_file_name}")
data = data.to_numpy(dtype=float)
# data = open(f"{current_wd}/{matlab_file_name}")
# data = data.readlines()
length = len(data)

# # convert to numpy matrix
# arr_data = data.to_numpy()
# # print(arr_data[0])
# print(arr_data[1])

# get the maximum size of each dataset
sizes = []
for list in data:
    # sizes.append(len(ast.literal_eval(list)))
    sizes.append(len(list))
maximum = max(sizes)
minimum = min(sizes)
print(f"Minimum = {minimum}")

# Now set out plotting parameters

time = np.linspace(t0, time_stop, minimum)

X = sci.odeint(dXdt, X0, time)

# Now plot the oscillations in time

x,y = X.T

# specify the plot style

plt.style.use("ggplot")

# for plotting both molecule numbers
x_mol = x#*omega
y_mol = y#*omega

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.plot(time, x_mol, '-r', label='x')
for i in range(0, length-1):
    # need to convert string to list
    # arr = ast.literal_eval(data[i])
    arr = data[i]
    arr = arr[:minimum]
    if i % 2 == 0:
        colour = 'r'
    elif i % 2 == 1:
        colour = 'b'
    ax1.scatter(time, arr, s=1, c=colour)

ax1.plot(time, y_mol, '-b', label='y')
# ax1.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Damped Oscillations in the Brusselator')
plt.show()

# for plotting the phase portrait
fig2 = plt.figure()
ax2 = fig2.add_subplot()
# plot SSA sample first as second plot will overlay it
y1 = []
y2 = []
for i in range(0, length-1):
    # need to convert string to list
    # arr = ast.literal_eval(data[i])
    arr = data[i]
    arr = arr[:minimum]
    if i % 2 == 0:
        y1.append(arr)
    elif i % 2 == 1:
        y2.append(arr)
mini = min(len(y1), len(y2))
for k in range(0, mini-1):
    ax2.scatter(y1[k], y2[k], s=1, c='r', label='pcLNA')
# ax2.plot(arr_data[0], arr_data[1], '-r', label='SSA')
ax2.plot(x_mol, y_mol, '-b', label='ODE')
# ax2.legend(loc='upper right')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Damped Oscillations in the Brusselator')
plt.show()