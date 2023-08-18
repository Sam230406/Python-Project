import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# Main function where the computational grid is created and the transfinite
# interpolation function is called and then the physical grid is plotted
def main():
    a = np.linspace(0, 1, 101)
    b = np.linspace(0, 1, 81)
    zeta, eta = np.meshgrid(a, b)
    D = 1
    x, y = Interpfunc(a, b, D)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(zeta, eta, 'o', c='y')
    ax1.grid()
    ax2.plot(x, y, 'o', c='y')
    ax2.grid()

# This function takes the x and y coordinated of all points in the
# computational plane, along with body diameter Dcand computes the
# corresponding corrdinates of the points in the physical plane. This function
# then returns these coordinates to "main()"

def Interpfunc(zeta, eta, D):
    grid_pts = len(zeta)*len(eta)
    x = np.zeros(grid_pts)
    y = np.zeros(grid_pts)
    k = 0
    for i in range(len(zeta)):
        for j in range(len(eta)):
            x[k] = ((1-zeta[i])*(0.5*D + 9.5*D*eta[j])) + ((1-eta[j])*(0.5*D*cos(2*pi*zeta[i])))\
                + (eta[j]*10*D*cos(2*pi*zeta[i])) - ((1-zeta[i])*(1-eta[j])*0.5*D) -\
                (eta[j]*(1-zeta[i])*10*D)
            y[k] = (zeta[i]*(0.5*D + 9.5*D*eta[j])) + ((1-eta[j])*(0.5*D*sin(2*pi*zeta[i])))\
                + (eta[j]*10*D*sin(2*pi*zeta[i])) - \
                ((1-eta[j])*zeta[i]*0.5*D) - (eta[j]*zeta[i]*10*D)
            k += 1
    return(x, y)


main()
