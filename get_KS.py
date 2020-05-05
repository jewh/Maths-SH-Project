
# Python file to calculate the KS distance between distributions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as st
import scipy.integrate as sci
import ast
import cmath
import matplotlib.gridspec as gridspec


def get_list_of_string(string, length):
    out = []
    for i in range(0, length):
        out[i] = string
    return out

# Define a function that reads a list of lists and returns the columns

def get_time_slices(list_of_lists):
    # list_of_lists = np.array(list_of_lists)
    # list_of_lists = list_of_lists.transpose()
    # return list(list_of_lists)
    # list of lists for each variable
    time_slices_x1 = []
    time_slices_x2 = []
    # calculate the maxiumum

    max = []
    for list in list_of_lists:
        max.append(len(list))
    minimum = min(max)
    # assumes input where each set of two lists is a repeat
    for i in range(0, minimum):
        sub_list_x1 = []
        sub_list_x2 = []
        for j in range(0, len(list_of_lists)):
            list = list_of_lists[j]
            elem = list[i]
            # sort if odd or even
            if j % 2 == 0:
                sub_list_x1.append(elem)
            elif j % 2 == 1:
                sub_list_x2.append(elem)
        time_slices_x1.append(sub_list_x1)
        time_slices_x2.append(sub_list_x2)

    return time_slices_x1, time_slices_x2

def get_time(list_of_lists):
    m = []
    for list in list_of_lists:
        m.append(len(list))
    return min(m)

def get_significance_level(threshold, m, n):
    m = float(m)
    n = float(n)
    out = np.sqrt(-np.log(threshold/2.0)*((1 + n/m)/2.0))/np.sqrt(n)
    return out

def remove_nan(list):
    out = [item for item in list if str(item) != 'nan']
    return out

# define the brusselator

def dXdt(x, t=0):
    dxdt = np.array([a + x[1]*x[0]**2 - b*x[0] - x[0], b*x[0] - x[1]*x[0]**2])
    return dxdt

def eigenvalues(a,b):
    #returns brusselator eigenvalues
    a = float(a)
    b = float(b)
    l1 = 0.5*(b -1.0 + a**2 + cmath.sqrt((b - 1.0 + a**2)**2 - 4.0*a**2))
    l2 = 0.5 * (b - 1.0 + a ** 2 - cmath.sqrt((b - 1.0 + a ** 2) ** 2 - 4.0 * a ** 2))
    return l1, l2

def jacobian(a, b):
    # calculates the jacobian at the point (a, b/a)
    r1 = [b-1, a**2]
    r2 = [-b, -a**2]
    return [r1, r2]

def distance(vector):
    length = len(vector)
    out = 0.0
    for i in range(0, length):
        vector[i] = float(vector[i])
        out += vector[i]**2
    out = np.sqrt(out)
    return out

def get_basis(matrix):
    # get an orthonormal basis for the two vectors defined by
    eigen = np.linalg.eig(matrix)[1]
    eigen = eigen.transpose()
    # as conjugate, just consider top eigenvector
    eigen = eigen[0]
    v1 = [np.real(eigen[0]), np.real(eigen[1])]
    v2 = [np.imag(eigen[0]), np.imag(eigen[1])]
    # Now orthnormalise
    u1 = v1/distance(v1)
    u2 = v2 - (np.dot(v2, u1)/np.dot(u1, u1))*u1
    u2 = u2/distance(u2)
    return u1, u2

def get_phase(vector):
    if vector[0] > 0:
        return np.arccos(vector[1]/distance(vector))
    elif vector[0] <= 0:
        return np.pi + np.arccos(-vector[1]/distance(vector))


def get_phases(list_of_two_lists):
    # assume that the length equal
    length = len(list_of_two_lists[0])
    out = []
    for i in range(0, length):
        X = list_of_two_lists
        vector = [X[0][i], X[1][i]]
        phi = get_phase(vector)
        out.append(phi)
    return out

def between(query, lower, upper):
    if query >= lower and query <= upper:
        return True
    else:
        return False

def calculate_phase(list_of_phasecoords):
    # input is list of phase coordinate calculated as above
    length = len(list_of_phasecoords)
    output = np.zeros(length, dtype=float)
    # initialise
    phi = list_of_phasecoords[0]
    for i in range(0, length):
        output[i] = phi
        s_tilda = list_of_phasecoords[i]
        for q in range(-10,11):
            if between(phi - 2.0*np.pi*q, 0, 2.0*np.pi):
                q1 = 2.0*q*np.pi + s_tilda - phi
                q2 = 2.0*(q-1.0)*np.pi + s_tilda - phi
                q3 = 2.0*(q+1.0)*np.pi + s_tilda - phi
                if between(q1, 0, np.pi):
                    phi = phi + q1
                    break
                elif between(q2, 0, np.pi):
                    phi = phi + q2

                    break
                elif between(q3, 0, np.pi):
                    phi = phi + q3

                    break
    # now subtract initial value so starts at 0
    base = output[0]
    for k in range(0, length):
        output[k] = output[k] - base
    return output

def find_min(listlike, integer):
    # return the index of the element closest to the value in the list
    length = len(listlike)
    diff = []
    for i in range(0, length):
        df = abs(listlike[i]-integer)
        diff.append(df)
    minimum = min(diff)
    # now get the index
    for j in range(0, length):
        if diff[j] == minimum:
            return j

# define a function that takes as input a trajectory of x and y, as well as the phases for these, and returns the
# trajectory updated to remove points with equal phase

def get_set_phase(x,y,phase):
    # now phase should be same length as x and y
    # so
    length = len(phase)
    ux = []
    uy = []
    up = []
    max_int = round(phase[length-1])
    max_int = int(max_int)
    mins = []
    for i in range(0, max_int):
        dp = find_min(phase,i)
        mins.append(dp)
    # print(len(mins))
    for index in mins:
        ux.append(x[index])
        uy.append(y[index])
        up.append(phase[index])
    return ux, uy, up


def get_phase_slices(list_of_lists, phases, multiple):
    # returns the list of x and y at specified multiples of pi
    out = []
    conv = 0
    for i in range(0, len(list_of_lists)-1):
        if i % 2 == 1:
            continue
        else:
            phase = phases[conv]/(multiple*np.pi)
            # print(phase[0:3])
            xs = list_of_lists[i]
            ys = list_of_lists[i+1]
            # max_int = int(phase[len(phase)-1])
            # print(f"fed = {max_int}")
            sete = get_set_phase(xs, ys, phase)
            xi = sete[0]
            yi = sete[1]
            # phase = list(phase)
            # # print(phase)
            # for value in phase:
            #     for j in range(0, len(phase)):
            #         if phase[j] == value:
            #             x.append(xi[j])
            #             y.append(yi[j])
            out.append(xi)
            out.append(yi)
            conv += 1
    # Now re-process out into list of phase slices for either variable
    out = [x for x in out if x != []]
    return out

def get_xticks(length, multiple):
    # returns a list that can be used to label plots with phase
    out = []
    period = 1.0/multiple
    if period > 0:
        period = int(period)
        counter = 0
        entry = counter
        for i in range(0, length):
            if i % period == 0:
                out.append(entry)
                counter += 1
                entry = f"{counter}$\pi$"
            else:
                out.append('')
    else:
        multiple = int(multiple)
        counter = 0
        entry = counter
        for i in range(0, length):
            if i % multiple == 0:
                out.append(entry)
                counter += multiple
                entry = f"{counter}$\pi$"
    return out

#_______________________________________________________________________________________________________________________
# First load in the data
current_wd = os.path.dirname(os.path.realpath(__file__))

SSA_filename = "Gillespie.csv"
pcLNA_filename = "lna1.5Omega300tend60.csv"
# Specify the parameters used for the simulation
ssa_omega = 300.0
a = 1
b = 1.5
x0 = 0.75
y0 = 2.0
t0 = 0
time_stop = 60

# Get the basis around the steady state
steady_state = [a, b/a]
steady_state = np.array(steady_state)
R = get_basis(jacobian(a, b))
R = np.array(R)

# Now load in the data
gillespie = pd.read_csv(f"{current_wd}/{SSA_filename}")
gillespie = gillespie.to_numpy(dtype=float)
gillespie = gillespie/ssa_omega
pcLNA = pd.read_csv(f"{current_wd}/{pcLNA_filename}")
pcLNA = pcLNA.to_numpy(dtype=float)
# Below just from my home-cooked LNA files
# pcLNA = pcLNA.transpose()
# pcLNA = np.delete(pcLNA, 0, 0)

# for string in pclna:
#     l = ast.literal_eval(string)
#     pcLNA.append(l)

# must convert each trajectory into phase, then create an array to store phase coordinates
phase_coords_LNA = []
for i in range(0, len(pcLNA)):
    x1 = []
    x2 = []
    if i % 2 == 1:
        continue
    else:
        # assuming that the length of both trajectories is equal
        for j in range(0, len(pcLNA[i])):
            x = pcLNA[i][j]
            y = pcLNA[i+1][j]
            Xi = np.array([x, y])
            Y = np.matmul(R.transpose(), Xi-steady_state)
            # so should give us a 2x1 matrix in return
            x1.append(Y[0])
            x2.append(Y[1])
    X = [x1, x2]
    ph = get_phases(X)
    phcoord = calculate_phase(ph)
    # phcoord = phcoord = phcoord[0]
    phase_coords_LNA.append(phcoord)

# Now same for the gillespie trajectories
phase_coords_gill = []
for i in range(0, len(gillespie)):
    if i % 2 == 1:
        continue
    else:
        x1 = []
        x2 = []
        x = list(gillespie[i])
        y = list(gillespie[i+1])
        x = remove_nan(x)
        y = remove_nan(y)
        # assuming that the length of both trajectories is equal
        for j in range(0, len(x)):
            xp = x[j]
            yp = y[j]
            Xi = np.array([xp, yp])
            Y = np.matmul(R.transpose(), Xi-steady_state)
            # so should give us a 2x1 matrix in return
            x1.append(Y[0])
            x2.append(Y[1])
    X = [x1, x2]
    ph = get_phases(X)
    phcoord = calculate_phase(ph)
    # phcoord = phcoord - phcoord[0]
    phase_coords_gill.append(phcoord)

print(f"LNA Tau = {0.5*phase_coords_LNA[0][len(phase_coords_LNA[0])-1]/np.pi}")
print(f"Gill Tau = {0.5*phase_coords_gill[0][len(phase_coords_gill[0])-1]/np.pi}")
# print(phase_coords_gill[0])
# print(phase_coords_LNA[0])
# for i in range(0, len(phase_coords_gill)):
#     x = np.arange(0, len(phase_coords_gill[i]))
#     plt.scatter(x, phase_coords_gill[i], s=1, c='r')
# for j in range(0, len(phase_coords_LNA)):
#     y = np.arange(0, len(phase_coords_LNA[j]))
#     plt.scatter(y, phase_coords_LNA[j], s=1, c='b')
# plt.show()
minimum = 1000

X0 = np.array([x0, y0])
time = np.linspace(t0, time_stop, minimum)
X = sci.odeint(dXdt, X0, time)
x,y = X.T
Y = [x, y]
px = []
py = []
for i in range(0, minimum):
    xi = x[i]
    yi = y[i]
    Xi = np.array([xi, yi])
    Y = np.matmul(R.transpose(), Xi - steady_state)
    px.append(Y[0])
    py.append(Y[1])
Y = [px, py]

phis = get_phases(Y)
phic = calculate_phase(phis)
print(f"ODE Tau = {0.5*phic[len(phic)-1]/np.pi}")

# plt.plot(px, py, 'b')
# plt.plot(x, y, 'r')
multiple = 0.5
gill = get_phase_slices(gillespie, phase_coords_gill, multiple)
LNA = get_phase_slices(pcLNA, phase_coords_LNA, multiple)

# Now calculate the KS distance between each
# First get the phase slices into true phase slices
p = get_time_slices(LNA)
ssa = get_time_slices(gill)
#
# print(p[0][:3])
# print(ssa[0][:3])

# create list to store the KS distances in
ks_distances_x1 = []
ks_distances_x2 = []

minimum = min([get_time(p), get_time(ssa)]) # subtract a number so that KS=1 not in plot

# Now loop and get the ks distance for each time slice
for time in range(0, minimum):
    # calculate the stuff for x1
    x1_pclnatime = p[0][time]
    x1_ssatime = ssa[0][time]
    ks1 = st.ks_2samp(x1_pclnatime, x1_ssatime)
    ks_distances_x1.append(ks1[0])

    # now for x2
    x2_pclnatime = remove_nan(p[1][time])
    x2_ssatime = remove_nan(ssa[1][time])
    ks2 = st.ks_2samp(x2_pclnatime, x2_ssatime)
    ks_distances_x2.append(ks2[0])

# Calculate the significance level
u = []
u.append(get_time(p[0]))
u.append(get_time(p[1]))
m = min(u)

v = []
v.append(get_time(ssa[0]))
v.append(get_time(ssa[1]))
n = min(v)

threshold = 0.01
sig_level = get_significance_level(threshold, m, n)

# Now calculate the ODE trajectory
resolution = 1000
X0 = np.array([x0, y0])
time = np.linspace(t0, time_stop, resolution)

X = sci.odeint(dXdt, X0, time)
# P = []
# phi = []
# for i in range(0, minimum):
#     p = np.matmul(R.transpose(), X[i]-steady_state)
#     P.append(p)
#     phi.append(get_phase(p))
# P = np.array(P)

# Can choose to plot the projection of the phase portrait or the phase portrait itself
x,y = X.T

# Now combine into one large figure
plt.style.use("ggplot")

fig = gridspec.GridSpec(9, 3)
ax1 = plt.subplot(fig[1:4, :-1])
ax2 = plt.subplot(fig[6:, :-1])
ax3 = plt.subplot(fig[1:, -1])

for i in range(0, min([len(gillespie), len(pcLNA)])):
    if i % 2 == 0:
        ax2.scatter(pcLNA[i], pcLNA[i+1], s=1, c='g')
        ax1.scatter(gillespie[i], gillespie[i+1], s=1, c='r')

ax1.plot(x, y, 'b')
ax2.plot(x, y, 'b')
# Set the axes to the same limits
xlim = 2.5
ylim = 4.0
ax1.set_ylim([-0.1, ylim])
ax1.set_xlim([-0.1, xlim])
ax2.set_ylim([-0.1, ylim])
ax2.set_xlim([-0.1, xlim])
# Now label the plot
ax1.set_title("SSA", fontsize=10) # loc ='left'
ax2.set_title("pcLNA", fontsize=10)

ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")


# Now plot the KS distances
x = np.arange(0, minimum)
ticks = get_xticks(minimum, multiple)
ax3.barh(x-0.1, width=ks_distances_x1, height=0.2, color='#00aa00', align='center', tick_label=ticks)   #00aa00
ax3.barh(x+0.1, width=ks_distances_x2, height=0.2, color='#348abdff', align='center', tick_label=ticks)
ax3.set_xlim([0, 0.5])
# ax3.set_ylim([0, 3])
# ax3.invert_yaxis()
ax3.invert_xaxis()
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()

ax3.set_xlabel("D")
ax3.set_ylabel("$\phi$")
# get the ticks for ax3


# Add in the significance level
ax3.vlines(sig_level, colors='k', ymin=0, ymax=minimum, linestyles='solid', lw=1.0)
ax3.set_title("KS Distances", fontsize=10)
plt.suptitle(f"SSA vs pcLNA \na={a}, b={b}, $\Omega$={int(ssa_omega)}, tmax=60", fontsize=14)
# plt.title(f"a = {a} b = {b} $\Omega$ = {int(ssa_omega)}", fontsize=10)
plt.tight_layout()
plt.show()

# Now something just to check that the phase works as we hope
fig2, (ax4, ax5) = plt.subplots(1,2)

# p = get_time_slices(pcLNA)
# ssa = get_time_slices(gillespie)

for i in range(0, len(p)):
    if i % 2 == 1:
        continue
    else:
        x = p[i]
        y = p[i+1]
        ax5.scatter(x,y,s=1,c='g')
for i in range(0, len(ssa)):
    if i % 2 ==1:
        continue
    else:
        x = ssa[i]
        y = ssa[i+1]
        ax4.scatter(x,y,s=1,c='r')
ax4.set_title("SSA", fontsize=10)
ax5.set_title("pcLNA", fontsize=10)
ax4.set_ylabel("y")
ax4.set_xlabel("x")
ax5.set_xlabel("x")
plt.suptitle(f"Points with phase = {multiple}k$\pi$", fontsize=14)
plt.show()


plt.close()
# Now plot a lil something
length = len(gillespie[0])

x = gillespie[0]
y = gillespie[1]

t = np.linspace(0, time_stop, length)
plt.scatter(t, x, color='b', s=1)
plt.scatter(t, y, color='r', s=1)
plt.title("SSA Simulation of the Brusselator", fontsize=14)


plt.show()

