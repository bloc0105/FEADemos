import numpy as num
from matplotlib import pyplot as plott
import sympy as sym


'''
Solves the equation \nabla u=f(x,y) using the Finite Element method. 

This Finite Element code uses symbollic math instead of numeric math by implementing
the sympy library. 

This Code solves for the functional of the equation then establishes 
boundary conditions and solves.

Most of the concepts for this code were taken from The Finite Element for Scientists
and Engineers, ISBN 0-471-37078-9.  As such, I have enumerated the steps of the finite element
method as labeled in that book, within the comments of the program. Unlike the book, however, this 
example is in two dimensions instead of just one. The functional for a two-dimensional
differential equation problem was taken from Partial Differential Equations for Scientists and 
Engineers, ISBN 0-486-67620-X.    
'''

#The boundary goes from 0 to 1.  Pick the number of grid points to increase or decrease 
#the accuracy (but also computation) of the algorithm. 

low_boundary = 0 # The lower boundary of the graph
high_boundary = 1 # The upper boundary of the graph
grid_points = 3 # The number of nodes that exist along the axes of the boundary (inclusive)

#1. Discretize the system.
number_of_divisions = (high_boundary - low_boundary) * grid_points #each quadrant will have the same number of nodes in it

x_range = num.linspace(low_boundary,high_boundary,number_of_divisions)
y_range = num.linspace(low_boundary,high_boundary,number_of_divisions)

element_range = number_of_divisions - 1


XGrid,YGrid = num.meshgrid(x_range, y_range) #Discretize the space into rectangular elements.

#This labels all the "nodes" in the system.  Each node is a sympy variable that will be solved
z_matrix = [[sym.symbols('z_x' + str(counterx) + '_y' + str(countery)) for counterx in range(len(XGrid))] for countery in range(len(XGrid[0]))]

#Variables to be used in the calculation.  
x, y, f= sym.symbols('x y f')
x0,y0,y1,x1,phi0,phix,phiy = sym.symbols('x_0 y_0 y_1 x_1 phi_0 phi_x phi_y')

#2. Set up The Element Parameters

#This creates the template element by approximating it as a plane.  
m = (phix - phi0)/(x1 - x0)
n = (phiy - phi0)/(y1 - y0)
b = phi0 - m * x0 - n * y0

phi_final = m * x + n * y + b #Equation for the plane

f = -1  # This is the equation being solved

#Solve for the functional of the system. 
phi = (sym.diff(phi_final,x))**2  +  (sym.diff(phi_final,y))**2 + 2 * f * phi_final
phi_integral = sym.integrate(sym.integrate(phi,(x,x0,x1)),(y,y0,y1))

#Minimize the functional
phi_diff_0 = sym.diff(phi_integral, phi0) #Minimum at the point (x_0,y_0)
phi_diff_x = sym.diff(phi_integral, phix) #Minimum at the point (x_1,y_0)
phi_diff_y = sym.diff(phi_integral, phiy) #Minimum at the point (x_0,y_1)

#3. Compute the Element Matrices. 
var_list = []
solution_matrices = []
solution_equations = []
zero_vals = []

#Each set of three points forms a right triangle. The triangles are solved and made into a matrix. 
for xcounter in range(number_of_divisions):
    for ycounter in range(number_of_divisions):

        #If not on an upper boundary, Create a trianlge going in the +x, +y direction
        if xcounter < element_range and ycounter < element_range:
            solution_list = []
            x_max = XGrid[ycounter][xcounter + 1]
            x_min = XGrid[ycounter][xcounter]

            y_max = YGrid[ycounter +1][xcounter]
            y_min = YGrid[ycounter][xcounter]

            z_y = z_matrix[ycounter + 1][xcounter]
            z_x = z_matrix[ycounter][xcounter + 1]
            z_0 = z_matrix[ycounter][xcounter]


            phi_subs_0 = phi_diff_0.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_x = phi_diff_x.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_y = phi_diff_y.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)

            solution_list.append(phi_subs_0)
            solution_list.append(phi_subs_x)
            solution_list.append(phi_subs_y)
            var_list.append([z_0,z_x,z_y])

            solution_matrices.append(sym.linear_eq_to_matrix(solution_list,[z_0,z_x,z_y]))
            solution_equations.append(solution_list)

        #This is used to establish boundary conditions
        else:
            zero_vals.append(z_matrix[ycounter][xcounter])
         
        #If not on a lower boundary, make a right triangle in the -x -y direction. 
        if xcounter > 0 and ycounter > 0 : 
            solution_list = []
            x_max = XGrid[ycounter][xcounter - 1]
            x_min = XGrid[ycounter][xcounter]
 
            y_max = YGrid[ycounter - 1][xcounter]
            y_min = YGrid[ycounter][xcounter]
 
            z_y = z_matrix[ycounter - 1][xcounter]
            z_x = z_matrix[ycounter][xcounter - 1]
            z_0 = z_matrix[ycounter][xcounter]
 
 
            phi_subs_0 = phi_diff_0.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_x = phi_diff_x.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_y = phi_diff_y.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
 
            solution_list.append(phi_subs_0)
            solution_list.append(phi_subs_x)
            solution_list.append(phi_subs_y)
            var_list.append([z_0,z_x,z_y])
 
            solution_matrices.append(sym.linear_eq_to_matrix(solution_list,[z_0,z_x,z_y]))
            solution_equations.append(solution_list)
            

#4.  Assemble the element equations into a global matrix. 
#This sets of a global system of equations and the list of unknown variables to be solved
z_vars = [z_matrix[countery][counterx] for countery  in range(len(z_matrix)) for counterx in range(len(z_matrix[0]))]
equation_system = [0 for countery  in range(len(z_matrix)) for counterx in range(len(z_matrix[0]))]

#This loop assembles each of the individual element equations into the global list of equations.
for solution_counter in range(len(solution_equations)):
    for eq_counter in range(len(solution_equations[solution_counter])):

        equation_system[z_vars.index(var_list[solution_counter][eq_counter])] += solution_equations[solution_counter][eq_counter]
print(sym.latex(sym.Matrix(equation_system)))
#Make the equations into a matrix to make them easier to solve. 
#5. Impose Boundary conditions. Zero at x = 1 and y = 1. This replaces all of the nodes in those positions with zeros.
replacement_eq = []
for equation in equation_system:
    eq = 0
    for each_var in zero_vals:
        eq = equation.subs(each_var,0)
    replacement_eq.append(eq)

# print(sym.latex(sym.Matrix(replacement_eq)))


# Conditions where the equation already solves to zero are a problem, because 0 = 0 will not evaluate
# This replaces the boundary condition and sets a specific variable to 0 so it can be solved. 
boundary_eq_with_result = []

for z_index,variable in enumerate(z_vars):
    if variable not in zero_vals:
          
        boundary_eq_with_result.append(replacement_eq[z_index])
    else:
        boundary_eq_with_result.append(sym.Eq(variable,0))

# print(sym.latex(sym.Matrix(boundary_eq_with_result)))

#6. Solve the System. Sympy linsolve makes short work of that.   
resultset = sym.linsolve(boundary_eq_with_result,z_vars)
print(sym.latex(resultset))

#7 Use the computed results to determine desired results. 
#In most FEA solutions, this would be stresses or fluid flow, but in this case, it's just the Z-Values.
result_vals = [resultset.args[0][counter] for counter in range(len(z_vars))]

result_grid = num.zeros([number_of_divisions,number_of_divisions])

#Make a meshgrid with the Z-Values in it
varcount = 0
for ycounter in range(len(z_matrix)):
    for xcounter in range(len(z_matrix[0])):
        result_grid[ycounter][xcounter] += result_vals[z_vars.index(z_matrix[ycounter][xcounter])]
        varcount += 1

#Plot the results
plott.contourf(XGrid,YGrid,result_grid, 150)
plott.colorbar()
plott.show()
