

import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import gillespy2
from gillespy2.solvers.numpy.basic_ode_solver import BasicODESolver

a = 1.0 # Fix this at 1 for ease
b = 1.7 # Vary this one <2 for damped oscillations, >=2 for sustained
x0 = 1
y0 = 2 # keeping same ratio of x0:y0 as in Giorgo's code
omega = 1000

# Define the system vector

X0 = np.array([x0, y0])

# Describe its evolution


def dXdt(x,t=0):
    dxdt = np.array([a + x[1]*x[0]**2 - b*x[0] - x[0], b*x[0] - x[1]*x[0]**2])
    return dxdt


# Define the model as Gillespie class now
# (Code adapted from https://github.com/GillesPy2/GillesPy2/blob/master/examples/BasicExamples/Brusselator.ipynb)

class Brusselator(gillespy2.Model):
    def __init__(self, parameter_values=None):
        gillespy2.Model.__init__(self, name="Brusselator", volume=omega)
        # Set timespan of model
        self.timespan(np.linspace(0, 30000, num=50))

        # List all species in the brusselator, and thus all reactions
        A = gillespy2.Species(name="A", initial_value=omega)  # If the concentrations of A and B are large then this is
        B = gillespy2.Species(name="B", initial_value=omega)   # analogous to the ODE system
        C = gillespy2.Species(name="C", initial_value=0)
        D = gillespy2.Species(name="D", initial_value=0)
        X = gillespy2.Species(name="X", initial_value=x0)
        Y = gillespy2.Species(name="Y", initial_value=y0)
        self.add_species([A, B, C, D, X, Y])

        # Parameters (rates)
        rate1 = gillespy2.Parameter(name="rate1", expression=a)
        rate2 = gillespy2.Parameter(name="rate2", expression=1.0)
        rate3 = gillespy2.Parameter(name="rate3", expression=1.0)
        rate4 = gillespy2.Parameter(name="rate4", expression=b)
        self.add_parameter([rate1, rate2, rate3, rate4])

        # Reactions
        reaction1 = gillespy2.Reaction(name="reaction1",
                                       reactants={A: 1},
                                       products={X: 1, A: 1},
                                       propensity_function="rate1")
        reaction2 = gillespy2.Reaction(name="reaction2",
                                       reactants={B: 1, X: 1},
                                       products={Y: 1, C: 1, B: 1},
                                       propensity_function="rate2 * X")
        reaction3 = gillespy2.Reaction(name="reaction3",
                                       reactants={X: 2, Y: 1},
                                       products={X: 3},
                                       propensity_function="rate3 * Y * X * (X - 1) / 2")
        reaction4 = gillespy2.Reaction(name="reaction4",
                                       reactants={X: 1},
                                       products={D: 1},
                                       propensity_function="rate4 * X")
        self.add_reaction([reaction1, reaction2, reaction3, reaction4])

        # Set list of species that should be plotted
        self.species_to_plot = ["X", "Y"]



# Now generate time series for ODE solution

time_stop = 100
time = np.linspace(0, time_stop, 1000)

# X = sci.odeint(dXdt, X0, time)
#
# # Now plot the oscillations in time
#
# x,y = X.T
# #
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax1.plot(time, x, '-r', label='x')
# ax1.plot(time, y, '-b', label='y')
# ax1.legend(loc='best')
# plt.xlabel('time')
# plt.ylabel('concentration')
# plt.title('Brusselator')



# Now plot the phase plane, with Gillespie trajectories
#
# Call the trajectories:

model = Brusselator()
results = gillespy2.GillesPySolver().run(model=model)
# ode_results = model.run(solver=BasicODESolver)
#
for i in range(0, 10):
    trajectory = results[i]
    plt.plot(trajectory['time'], trajectory['X'], 'r')
    plt.plot(trajectory['time'], trajectory['Y'],  'b')
plt.legend(loc='best')
plt.show()

# ax2 = fig.add_subplot(2,1,2)
plt.plot(x, y, '-b', label='Brusselator')
for i in range(0, 10):
    trajectory = results[i]
    plt.plot(trajectory['X'], trajectory['Y'], '-r')
plt.legend(loc='best')
plt.title('Phase portrait')
plt.xlabel('x')
plt.ylabel('y')
plt.show()





