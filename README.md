# Python-Project
Implementation of finite differences CFD schemes in Python
## for 1.py ( stream function equation using central differencing and point-jacobi method )
1. In the function "main()", change the value of the variable "initial_guess" to anyone of [100,150,200].
2. Run the program, which will then generate the .csv files for converged solution at x = 0,1,2,3 as well
   as the contour plot of the final converged solution.

## for 2.py ( Transfinite Interpolation)
1. Open the .py file and set the working directory in the IDE
2. Run the program 
3. Check the saved image "TFI_Output.png" in the working directory folder.

## for 3.py ( Couette Flow using Crank Nicholson Scheme and FTCS scheme)
1. Open 3.py.py
2. In "main()" function, locate "time" variable.
3. Assign the value of time at which you want to see velocity profile.
4. Run the program and obtain the plots for velocity profile and error.

## for 4.py ( Stager-Warming Flux Vector Splitting) 
1. Set the working directory
2. Save the exact solution excel file as "exact_solution_shock_tube.xlsx" in the working directory.
3. Make a column for mach number in the excel file (not already present in given file).
4. Make the heading of mach number column as 'M'.
5. Populate the column as M = u/sqrt(1.4*287*temperature)
6. Run the .py file
7. A figure named "final_figure.png" will be saved in working directory.
