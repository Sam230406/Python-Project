import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''All the data for problem physics is defined in this function, along with the
   boundary conditions.
'''
def data():
    cp = 1005
    gamma = 1.4
    R = 287.1
    nu = 0.9
    p = np.zeros(103)
    T = np.zeros(103)
    u = np.zeros(103)
    M = np.zeros(103)
    rho = np.zeros(103)
    E = np.zeros(103)
    a = np.zeros(103)
    dx = 0.01
    p[:50] = 5*101325
    p[51:] = 1*101325
    T[:] = 300
    E[:] = (0.5*rho*u*u) + (p/(gamma-1))
    u[:] = 0
    a[:] = np.sqrt(gamma*R*T)
    M[:] = u/np.sqrt(gamma*R*T)
    rho[:] = p[:]/(R*T[:])
    t_req = 0.00075
    return(R,gamma,cp,p,T,u,M,rho,E,a,dx,t_req,nu)

''' Lambda max is calculated here for evaluating time step
'''
def get_lam_max(u, a):
    lam_max = np.max(np.abs(u[1:102]) + a[1:102])
    return lam_max

'''The time step value using lambda max is calculated here
'''
def get_dt(nu, dx, lam_max):
    dt = nu*dx/lam_max
    return dt

'''This function classifies the cases for each grid point using its current Mach
   number, and then calls the corresponding functions for assigning the flux
   vectors for the mass, momentum energy and energy terms.
'''
def assign_F(M,u,a,rho,p,E,gamma):    
    indices_case_1 = [i for i in range(len(M)) if M[i] <= -1]
    indices_case_2 = [i for i in range(len(M)) if M[i] > -1 and M[i]
                      <= 0]
    indices_case_3 = [i for i in range(len(M)) if M[i] > 0 and M[i]
                      <= 1]
    indices_case_4 = [i for i in range(len(M)) if M[i] > 1]
    F_plus_mass,F_minus_mass = F_mass(u, a, rho, p, E, gamma, indices_case_1, indices_case_2, indices_case_3, indices_case_4)
    F_plus_momentum,F_minus_momentum = F_momentum(u, a, rho, p, E, gamma, indices_case_1, indices_case_2, indices_case_3, indices_case_4)
    F_plus_energy,F_minus_energy = F_energy(u, a, rho, p, E, gamma, indices_case_1, indices_case_2, indices_case_3, indices_case_4)
    return(F_plus_mass,F_minus_mass,F_plus_momentum,F_minus_momentum,F_plus_energy,F_minus_energy)
    
                   
'''The plus and minus flux vector equations for mass are used here to create the
   flux vector.
'''
def F_mass(u,a,rho,p,E,gamma,indices_case_1,indices_case_2,indices_case_3,\
           indices_case_4):
    F_plus_1 = np.zeros(103)
    F_minus_1 = np.zeros(103)
    F_plus_1[indices_case_1] = 0
    F_plus_1[indices_case_2] = (rho[indices_case_2]*(u[indices_case_2] \
                                + a[indices_case_2])/(2*gamma))
    F_plus_1[indices_case_3] = ((gamma-1)*rho[indices_case_3]* \
                               u[indices_case_3]/gamma)+ (rho[indices_case_3]* \
                               (u[indices_case_3]+a[indices_case_3])/(2*gamma))
    F_plus_1[indices_case_4] = rho[indices_case_4]*u[indices_case_4]
    F_minus_1[indices_case_1] = rho[indices_case_1]*u[indices_case_1]
    F_minus_1[indices_case_2] = ((gamma-1)*rho[indices_case_2]* \
                               u[indices_case_2]/gamma)+ (rho[indices_case_2]* \
                               (u[indices_case_2]-a[indices_case_2])/(2*gamma))
    F_minus_1[indices_case_3] = (rho[indices_case_3]*(u[indices_case_3]- \
                                a[indices_case_3])/(2*gamma))
    F_minus_1[indices_case_4] = 0
    return(F_plus_1,F_minus_1)
    
'''The plus and minus flux vector equations for momentum are used here to create the
   flux vector.
'''
def F_momentum(u,a,rho,p,E,gamma,indices_case_1,indices_case_2,indices_case_3,\
               indices_case_4):
    F_plus_2 = np.zeros(103)
    F_minus_2 = np.zeros(103)
    F_plus_2[indices_case_1] = 0
    F_plus_2[indices_case_2] = (rho[indices_case_2]*(u[indices_case_2] 
                                + a[indices_case_2])/(2*gamma))* \
                                (u[indices_case_2] + a[indices_case_2])
    F_plus_2[indices_case_3] = ((gamma-1)/gamma)*(rho[indices_case_3]* 
                                u[indices_case_3]*u[indices_case_3])+ \
                                ((rho[indices_case_3]/(2*gamma))* 
                                 (u[indices_case_3] + a[indices_case_3])*  
                                 (u[indices_case_3] + a[indices_case_3]))
    F_plus_2[indices_case_4] = rho[indices_case_4]*u[indices_case_4] \
                                *u[indices_case_4] + p[indices_case_4]
    F_minus_2[indices_case_1] = rho[indices_case_1]*u[indices_case_1] \
                                *u[indices_case_1] + p[indices_case_1]
    F_minus_2[indices_case_2] = ((gamma-1)/gamma)*(rho[indices_case_2]* 
                                u[indices_case_2]*u[indices_case_2])+ \
                                ((rho[indices_case_2]/(2*gamma))* 
                                 (u[indices_case_2] - a[indices_case_2])*  
                                 (u[indices_case_2] - a[indices_case_2]))
    F_minus_2[indices_case_3] = ((rho[indices_case_3]/(2*gamma))* 
                                 (u[indices_case_3] - a[indices_case_3])*  
                                 (u[indices_case_3] - a[indices_case_3]))
    F_minus_2[indices_case_4] = 0
    return(F_plus_2,F_minus_2)
                                
'''The plus and minus flux vector equations for energy are used here to create the
   flux vector.
'''
def F_energy(u,a,rho,p,E,gamma,indices_case_1,indices_case_2,indices_case_3,\
               indices_case_4):
    F_plus_3 = np.zeros(103)
    F_minus_3 = np.zeros(103)
    F_plus_3[indices_case_1] = 0
    F_plus_3[indices_case_2] = (rho[indices_case_2]*(u[indices_case_2] \
                                + a[indices_case_2])/(2*gamma)) \
                               *((0.5*u[indices_case_2]*u[indices_case_2]) \
                                +((a[indices_case_2]*a[indices_case_2])/  
                               (gamma-1)) +(a[indices_case_2]*
                                u[indices_case_2]))
    F_plus_3[indices_case_3] = ((gamma-1)/gamma)*(rho[indices_case_3]* 
                                u[indices_case_3])*(0.5*u[indices_case_3]* 
                                u[indices_case_3]) + (rho[indices_case_3]/  
                                (2*gamma))*(u[indices_case_3] + 
                                a[indices_case_3])*((0.5*u[indices_case_3]* 
                                u[indices_case_3]) +((a[indices_case_3]* 
                                a[indices_case_3])/(gamma-1))+ 
                                (a[indices_case_3]*u[indices_case_3]))
    F_plus_3[indices_case_4] = u[indices_case_4]*(E[indices_case_4] + 
                                p[indices_case_4])
    F_minus_3[indices_case_1] = u[indices_case_1]*(E[indices_case_1] + 
                                p[indices_case_1])
    F_minus_3[indices_case_2] = ((gamma-1)/gamma)*(rho[indices_case_2]* 
                                u[indices_case_2])*(0.5*u[indices_case_2]* 
                                u[indices_case_2]) + (rho[indices_case_2]/  
                                (2*gamma))*(u[indices_case_2] + 
                                a[indices_case_2])*((0.5*u[indices_case_2]* 
                                u[indices_case_2]) -((a[indices_case_2]* 
                                a[indices_case_2])/(gamma-1)) -
                                (a[indices_case_2]*u[indices_case_2]))
    F_minus_3[indices_case_3] = (rho[indices_case_3]/(2*gamma))* \
                                (u[indices_case_3] - a[indices_case_3])* \
                                ((0.5*u[indices_case_3]*u[indices_case_3]) + 
                                ((a[indices_case_3]* a[indices_case_3])/ 
                                (gamma-1))-(a[indices_case_3]* 
                                u[indices_case_3]))
    F_minus_3[indices_case_4] = 0
    return(F_plus_3,F_minus_3)
    
'''The upwind scheme is coded here, which is called repeatedly to solve for the 
   source vector
'''
def upwind_scheme(U, F_plus, F_minus, dt, dx):
    U[:] = U[:] - ((dt/dx)*(F_plus[1:102] - F_plus[:101])) - \
        ((dt/dx)*(F_minus[2:] - F_minus[1:102]))
    return U

def plot(T,p,M,u):
    x = np.linspace(0,1,103)
    file = pd.read_excel('exact_solution_shock_tube.xlsx',skiprows=1)
    x_exact = file['x']
    p_exact = file['P']
    plt.plot(x_exact,p_exact)
    plt.show()

''' This is the main driving function, which calculates the source vector using
    using the while loop, and calculates the final parameters in the problem 
    at the stipulated time.
'''
def main():
    R,gamma,cp,p,T,u,M,rho,E,a,dx,t_req,nu = data() 
    time = 0
    U_mass = rho*np.ones(103)
    U_momentum = rho*u*np.ones(103)
    U_energy = (0.5*rho*u*u) + (p/(gamma-1))
    while(time < t_req):        
        lam_max = get_lam_max(u, a) 
        dt = get_dt(nu, dx, lam_max)
        if((t_req - time)<dt):
            dt = t_req - time
        F_plus_mass,F_minus_mass,F_plus_momentum,F_minus_momentum,F_plus_energy,F_minus_energy = assign_F(M, u, a, rho, p, E, gamma)    
        U_mass[1:102] = upwind_scheme(U_mass[1:102], F_plus_mass, F_minus_mass, dt, dx)
        rho = U_mass
        U_momentum[1:102] = upwind_scheme(U_momentum[1:102], F_plus_momentum, F_minus_momentum, dt, dx)
        u = U_momentum/rho
        U_energy[1:102] = upwind_scheme(U_energy[1:102], F_plus_energy, F_minus_energy, dt, dx)
        E = U_energy
        p = (E - (0.5*rho*u*u))*(gamma-1)
        T = p/(rho*R)
        a = np.sqrt(gamma*R*T)
        M = u/a
        time = time + dt
    x = np.linspace(0,1,103)
    file = pd.read_excel('exact_solution_shock_tube.xlsx',skiprows=1)
    x_exact = file['x']
    p_exact = file['P']
    T_exact = file['T']
    M_exact = file['M']
    u_exact = file['u']
    fig,axs = plt.subplots(2,2)
    axs[0,0].plot(x_exact,T_exact,'r')
    axs[0,0].plot(x,T,'b')
    axs[0,0].legend(["Exact","Numerical"])
    axs[0,0].set_title("Temperature")
    axs[0,1].plot(x_exact,p_exact,'r')
    axs[0,1].plot(x,p,'b')
    axs[0,1].legend(["Exact","Numerical"])
    axs[0,1].set_title("Pressure")
    axs[1,0].plot(x_exact,u_exact,'r')
    axs[1,0].plot(x,u,'b')
    axs[1,0].legend(["Exact","Numerical"],loc='best')
    axs[1,0].set_title("Velocity")
    axs[1,1].plot(x_exact,M_exact,'r')
    axs[1,1].plot(x,M,'b')
    axs[1,1].legend(["Exact","Numerical"],loc='lower center')
    axs[1,1].set_title("Mach No.")
    fig.tight_layout(pad=1.0)
    fig.savefig("final_figure.png")
main()
