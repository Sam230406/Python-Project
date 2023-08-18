import numpy as np
import matplotlib.pyplot as plt


'''The parameters to evaluate the error function are specified here
'''
def err_fn(x):
    sign = np.ones_like(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    # Save the sign of x
    for i in range(len(x)):
        if(x[i] < 0):
            sign[i] = -1
    x = np.abs(x)
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return(sign*y)

'''The exact analytical solution of the problem is computed in this function.
   This function calls the error function for various terms of the solution
'''
def exact_soln(plate_vel, h, nu, y, t):
    eta = y/(2*(np.sqrt(nu*t)))
    eta_1 = h/(2*(np.sqrt(nu*t)))
    term_1 = err_fn(eta)
    term_2 = err_fn(2*eta_1-eta)
    term_3 = err_fn(2*eta_1+eta)
    term_4 = err_fn(4*eta_1-eta)
    term_5 = err_fn(4*eta_1+eta)
    term_6 = err_fn(6*eta_1-eta)
    u_t = plate_vel*(term_1-term_2+term_3-term_4+term_5-term_6)
    return u_t

'''The FTCS scheme is solved in this function and final velocity array is 
   returned
'''
def FTCS(beta, u_0, iters):
    u_ftcs = u_0
    for i in range(iters):
        u_ftcs[1:-1] = u_ftcs[1:-1] + (beta*(u_ftcs[2:] - 2*u_ftcs[1:-1] + u_ftcs[0:-2]))
    return -u_ftcs


'''The Crank-Nicholson scheme is solved for the problem in this function.
   This function populates the coefficient and the constant matrix, and then
   solves for the velocity array, and returns it to the calling function.
'''
def crank_nicholson(beta, u_0, plate_vel, iters):
    # populating the coeff. matrix
    size = len(u_0)-2
    u_cr = np.zeros_like(u_0)
    u_1 = u_0[1:-1]
    A = np.zeros((size, size))
    A[0, 0:2] = [-(1+2*beta), beta]
    A[size-1, -2:] = [beta, -(1+2*beta)]
    for i in np.arange(1, size-1, 1):
        A[i, i-1:i+2] = [beta, -(1+2*beta), beta]
    for i in range(iters):
        # populate the constant matrix
        B = constant_matrix(u_1, beta, size, plate_vel)
        u_2 = np.linalg.solve(A, B)
        u_1 = u_2
    u_cr[0] = plate_vel
    u_cr[-1] = 0
    u_cr[1:-1] = u_1
    return(1*-u_cr)

'''This function populates the constant matrix for AX=B, this is called by
   the "crank_nicholson" function.
'''
def constant_matrix(u_1, beta, size, plate_vel):
    B = np.zeros(size)
    B[0] = -u_1[0] - (2*plate_vel*beta) - beta*(u_1[1]-2*u_1[0])
    B[-1] = -u_1[-1] - beta*(-2*u_1[-1] + u_1[-2])
    for i in np.arange(1, size-1, 1):
        B[i] = -u_1[i] - beta*(u_1[i+1] - 2*u_1[i] + u_1[i-1])
    return B


'''The plotting routine is handled by this function.
'''
def plot_fn(time,y, u_analytic, u_cr, u_ftcs, error_cr, error_ftcs):
    plt.subplot(211)
    plt.plot(u_analytic, y, marker = "o")
    plt.plot(u_cr, y, marker = "x")
    plt.plot(u_ftcs,y, marker = "+")
    plt.xlim([0, 40])
    plt.ylim([0, 0.04])
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Vertical distance (m)")
    plt.legend(["Analytical", "Crank-Nicholson Scheme","FTCS Scheme"])
    plt.title("Velocity Profile at time= %s seconds" % time)
    plt.grid(True)
    plt.subplot(212)
    plt.subplots_adjust(top = 3)
    plt.plot(error_cr, y)
    plt.plot(error_ftcs, y)
    plt.xlabel("Relative Error (m/s)")
    plt.ylabel("Vertical distance (m)")
    plt.legend(["Error:Crank-Nicholson Scheme","Error: FTCS Scheme"])
    plt.title("Relative Error at time= %s seconds" % time)
    plt.grid(True)
    plt.savefig("output.png")


'''The main driving function is defined, which contains all physical data 
   values as well as error definitions.
'''
def main():
    h = 0.04
    nu = 0.000217
    plate_vel = -40
    time = 0.15
    y = np.linspace(0, h, 61)
    u_0 = np.zeros_like(y)
    dt = 0.001
    dy = h/(len(y)-1)
    iters = int(time/dt)
    beta = 0.5*nu*dt/(dy*dy)
    beta_ftcs = nu*dt/(dy*dy)
    u_0[0] = plate_vel
    u_analytic = exact_soln(plate_vel, h, nu, y, time)
    u_cr = crank_nicholson(beta, u_0, plate_vel, iters)
    u_ftcs = FTCS(beta_ftcs, u_0, iters)
    error_cr = np.zeros_like(y)
    error_ftcs = np.zeros_like(y)
    error_cr[1:-1] = (u_analytic[1:-1]-u_cr[1:-1])/u_analytic[1:-1]
    error_ftcs[1:-1] = (u_analytic[1:-1]-u_ftcs[1:-1])/u_analytic[1:-1]
    plot_fn(time,y, u_analytic, u_cr, u_ftcs, error_cr, error_ftcs)


main()
