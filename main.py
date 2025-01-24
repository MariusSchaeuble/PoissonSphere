import matplotlib
import numpy
import numpy as np
import sympy as sym
from Helpers import identifier, isCharacter
import math
from numpy import matrix, array, mean, std, max, linspace, ones, sin, cos, tan, arctan, pi, sqrt, exp, arcsin, arccos, arctan2, sinh, cosh, zeros, log, diag, linspace, arange
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xlabel, ylabel, legend, title, savefig, errorbar, grid
import scipy.optimize as opt
from GPII import *
pi = math.pi
import scipy

# Set the Matplotlib backend for interactivity
import matplotlib
matplotlib.use('Qt5Agg')

#hardcode the case of a sphere, use poissons formula

r = 1.5
gridFeinheit = round(20*r)
homogen = 0 #save runtime by ignoring the inhomogen case, 1 is true, rest is false

def g(y1, y2):
    #return sin(math.atan2(y1, y2)*2)
    #return sin((y1 * y2))
    #return 1
    return y1

def f(y1, y2):
    #return -10*y1
    #return 0
    return -1
    #return sin(y1*y2)

def abs2D(k1, k2):
    #print(sqrt(k1**2 + k2**2))
    return sqrt(k1**2 + k2**2)
    #return min([2*r, sqrt(k1 ** 2 + k2 ** 2)])# all occuring points are within the sphere, so abs>2r would be num. artefacts


def Phi(k1, k2):
    return -log(abs2D(k1, k2))/2/pi

def u(x1, x2):
    if abs2D(x1, x2) > 0.99*r:
        return g(x1, x2)
    else:
        intFeinheit = 10*gridFeinheit #only few actual x points, but these are presisely calculated by much more y points, to get to the convergence
        phi = linspace(0, 2 * pi, intFeinheit)  # end and start included!
        points = array([r * cos(phi), r * sin(phi)])
        # trapezregel trapezoidal rule for integrals:
        delta = r * (phi[1] - phi[0])

        def integrand(y1, y2):
            return g(y1, y2) / abs2D(x1 - y1, x2 - y2)**2

        integral1 = 0
        for i in range(len(points[0]) - 1):
            integral1 += delta * 0.5 * (integrand(points[0][i], points[1][i]) + integrand(points[0][i + 1], points[1][i + 1]))

        t1 = (r**2 - abs2D(x1, x2)**2)/pi/r/2*integral1
        if homogen != 1:
            #representation formula: second term for inhomogen sol/poisson
            intFeinheit = round(sqrt(gridFeinheit))*10 #also high resolution in y for few x points, but taking into account the n^2 runtime for a circle using sqrt
            phi = linspace(0, 2 * pi, intFeinheit)  # end and start included!
            rs = linspace(r/intFeinheit, r, intFeinheit)
            a, b = np.meshgrid(phi, rs)
            X1 = b*cos(a)
            X2 = b*sin(a)

            absx = abs2D(x1, x2)
            def integrand(y1, y2):
                return f(y1, y2)*r**2*( Phi((y1 - x1)/r, (y2 - x2)/r)  -  Phi(absx/r**2*(y1 - x1/absx**2*r**2), absx/r**2*(y2 - x2/absx**2*r**2)) )

            integral2 = 0
            deltar = rs[1] - rs[0]
            deltaphi = phi[1] - phi[0]
            for i in range(np.shape(X1)[0]):
                for j in range(np.shape(X1)[1]):
                    integral2 += deltar*deltaphi*b[i, j]*integrand(X1[i, j], X2[i, j])
            t2 = integral2
        else:
            t2 = 0
        return t1 + t2




# Create a grid
phi = linspace(0, 2*pi, gridFeinheit)
rs = linspace(r/(100*gridFeinheit), r, gridFeinheit)
a, b = np.meshgrid(phi, rs)
X1 = b*cos(a)
X2 = b*sin(a)
#X, Y as export values for plot, eval von u
Z = zeros(np.shape(X1))
for i in range(np.shape(X1)[0]):
    for j in range(np.shape(X1)[1]):
        Z[i, j] = u(X1[i, j], X2[i, j])
#Z = u(X, Y) is the intended effect, but forced to perform elementwiise


# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])  # Equal scaling for all axes

# Plot the surface
surface = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

# Adjust tick parameters
plt.tick_params(axis='both', labelsize=25)

# Add labels and color bar
ax.set_xlabel('x1', fontsize=25)
ax.set_ylabel('x2', fontsize=25)
ax.set_zlabel('u(x1, x2)', fontsize=25)

# Prevent axes from cutting into labels
ax.xaxis.labelpad = 20  # Padding for x-axis label
ax.yaxis.labelpad = 20  # Padding for y-axis label
ax.zaxis.labelpad = 20  # Padding for z-axis label



# Adjust color bar font size
cbar = fig.colorbar(surface)
cbar.ax.tick_params(labelsize=25)

# Show the interactive plot
plt.show()


