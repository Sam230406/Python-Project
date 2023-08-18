import numpy as np
import matplotlib.pyplot as plt
import time


def L2_norm_denr(zi_n_plus_one):
    """This function evaluates the L2 norm of the denominator at the (n+1)th
        iteration and then returns the same to the calling function.

    INPUTS: array of the grid stream-function at the (n+1)th iteration
    OUTPUTS: L2 norm of the denominator of ERROR, as defined in assignment
    """
    sum_norm = np.sum(np.square(zi_n_plus_one[1:-1, 1:-1]))
    l2_denr = np.sqrt(sum_norm)
    return l2_denr


def L2_norm_numr(zi_n_plus_one, zi_n):
    """This function evaluates the L2 norm of the numerator and then returns
        the same to the calling function.

    INPUTS: 1) array of the grid stream-function at the (n+1)th iteration
            2) array of the grid stream-function at the (n)th iteration
    OUTPUTS: L2 norm of the numerator of ERROR, as defined in assignment
    """
    sum_norm = np.sum(np.square(zi_n_plus_one[1:-1, 1:-1] - zi_n[1:-1, 1:-1]))
    l2_numr = np.sqrt(sum_norm)
    return l2_numr


def Pt_Jacobi(zi_n):
    """This function uses the Point Jacobi Method to evaluate the values of
        stream function at the (n+1)th iteration using the values of stream
        function in the grid at the (n)th iteration, using the central
        differencing scheme.

    INPUTS: array of the grid stream-function at the (n)th iteration.
    OUTPUTS: array of the grid stream-function at the (n+1)th iteration.
    """
    zi_n_plus_one = np.copy(zi_n)
    zi_n_plus_one[1:-1, 1:-1] = (zi_n[2:, 1:-1] + zi_n[:-2, 1:-1] +
                                 zi_n[1:-1, 2:] + zi_n[1:-1, :-2]) * 0.25
    return(zi_n_plus_one)


def main():
    """This is the main driving function which contains the hard-coded data
        such as the ERROR threshold, initial guess value, and boundary
        conditions on the grid. Here a "while" loop is used to continuously
        monitor the ERROR and stop when ERROR becomes less than ERROR
        threshold. Also the contour plot of streamlines is plotted here as well
        as the solution value at the required x-locations is outputted as .csv
        files.
    """
    error_threshold = 0.0001
    initial_guess = 100
    grid = np.ones((40, 30)) * initial_guess
    grid[:, 0] = 100
    grid[-1, :] = 100
    grid[0, :10] = 100
    grid[0, 11:] = 200
    grid[0:20, -1] = 200
    grid[21:, -1] = 100
    zi_n = np.copy(grid)
    zi_n_plus_one = Pt_Jacobi(zi_n)
    error = L2_norm_numr(zi_n_plus_one, zi_n)/L2_norm_denr(zi_n_plus_one)
    zi_n = zi_n_plus_one
    iteration_number = 1
    s = time.perf_counter()
    while(error > error_threshold):
        zi_n_plus_one = Pt_Jacobi(zi_n)
        error = L2_norm_numr(zi_n_plus_one, zi_n)/L2_norm_denr(zi_n_plus_one)
        zi_n = zi_n_plus_one
        iteration_number = iteration_number + 1
    time_taken = time.perf_counter() - s
    x_0_solution = zi_n[:, 0]
    x_0_solution.tofile("x_0_"+"initial_guess_" +
                        str(initial_guess)+".csv", sep=",")
    x_1_solution = zi_n[:, 9]
    x_1_solution.tofile("x_1_"+"initial_guess_" +
                        str(initial_guess)+".csv", sep=",")
    x_2_solution = zi_n[:, 19]
    x_2_solution.tofile("x_2_"+"initial_guess_" +
                        str(initial_guess)+".csv", sep=",")
    x_3_solution = zi_n[:, 29]
    x_3_solution.tofile("x_3_"+"initial_guess_" +
                        str(initial_guess)+".csv", sep=",")
    plt.contourf(zi_n)
    plt.colorbar()
    plt.title("Streamline contours for initial guess $\psi$ =" +
              str(initial_guess) + "\n, No. of iterations for acheiving" +
              " tolerance $10^{-4}$=" + str(iteration_number+1))
    plt.show()
    print(time_taken)


main()
