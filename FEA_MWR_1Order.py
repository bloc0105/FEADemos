import numpy as num
from matplotlib import pyplot as plott
import sympy as sym
from sympy.vector import CoordSys3D as coord3
from sympy.vector import Del 
from sympy.vector import Vector

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
grid_points = 10 # The number of nodes that exist along the axes of the boundary (inclusive)

R = coord3('R')
delop = Del()

x = R.x
y = R.y
#1. Discretize the system.
number_of_divisions = (high_boundary - low_boundary) * grid_points #each quadrant will have the same number of nodes in it

x_range = num.linspace(low_boundary,high_boundary,number_of_divisions)
y_range = num.linspace(low_boundary,high_boundary,number_of_divisions)

element_range = number_of_divisions - 1


XGrid,YGrid = num.meshgrid(x_range, y_range) #Discretize the space into rectangular elements.

#This labels all the "nodes" in the system.  Each node is a sympy variable that will be solved
z_matrix = [[sym.symbols('z_x' + str(counterx) + '_y' + str(countery)) for counterx in range(len(XGrid))] for countery in range(len(XGrid[0]))]

#Variables to be used in the calculation.  
x0,y0,y1,x1,phi0,phix,phiy= sym.symbols('x_0 y_0 y_1 x_1 phi_0 phi_x phi_y')

#2. Set up The Element Parameters

#This creates the template element by approximating it as a plane.  
m = (phix - phi0)/(x1 - x0)
n = (phiy - phi0)/(y1 - y0)
b = phi0 - m * x0 - n * y0

phi_final = m * x + n * y + b #Equation for the plane

phi_solve_vars = [phi0,phix,phiy]

phi_poly = sym.Poly(phi_final,phi_solve_vars)
# print(sym.latex(phi_poly.as_expr()))

interpolation_functions = [phi_poly.as_expr().coeff(counter) for counter in  phi_solve_vars]


print(sym.latex(sym.Matrix(interpolation_functions)))

# print(sym.latex(phi_final))
f = -1  # This is the equation being solved

phi_diff_2 = sym.diff(phi_final,x,x) + sym.diff(phi_final,y,y)
phi_diff_1 = sym.diff(phi_final,x) + sym.diff(phi_final,y)
v = sym.diff(phi_final,x) * R.i + sym.diff(phi_final,y) * R.j
# print('----------------------------------------------------------------------------')
# print(sym.latex(phi_diff_2))
# print(sym.latex(phi_diff_1))

residual = phi_diff_2 - f
q = R.i + R.j
# print(sym.latex(sym.Matrix(phi_solved)))

weighted_averages = []



# weighted_averages = [ func.subs(x,x0).subs(y,y0) * phi_diff_1.subs(x,x0).subs(y,y0) - sym.integrate(sym.integrate(phi_diff_1 * func,(x,x0,x1)),(y,y0,y1)) + sym.integrate(sym.integrate(f * func,(x,x0,x1)),(y,y0,y1)) for func in interpolation_functions]

# print(sym.latex(sym.Matrix(weighted_averages)))
# print('===================================')


#3. Compute the Element Matrices. 
var_list = []
solution_matrices = []
solution_equations = []
zero_vals = []

#Each set of three points forms a right triangle. The triangles are solved and made into a matrix. 
for xcounter in range(number_of_divisions):
    for ycounter in range(number_of_divisions):
        q =  Vector.zero
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
            
            for function in interpolation_functions:
#                 print('---------------------------------------------------------------------')
#                 print(sym.latex(sym.integrate(sym.integrate( f * function,(x,x0,x1)),(x,y0,y1))))
                w = function * (v.dot(q)) - sym.Integral(sym.Integral( v.dot(delop(function)),(x,x0,x1)),(y,y0,y1)) - sym.Integral(sym.Integral( f * function,(x,x0,x1)),(y,y0,y1))
                weighted_averages.append(w)
        
            phi_diff_0 = weighted_averages[0] #Minimum at the point (x_0,y_0)
            phi_diff_x = weighted_averages[1] #Minimum at the point (x_1,y_0)
            phi_diff_y = weighted_averages[2] #Minimum at the point (x_0,y_1)
    


            phi_subs_0 = phi_diff_0.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_x = phi_diff_x.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_y = phi_diff_y.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)

            solution_list.append(phi_subs_0.doit())
            solution_list.append(phi_subs_x.doit())
            solution_list.append(phi_subs_y.doit())
            var_list.append([z_0,z_x,z_y])

            solution_matrices.append(sym.linear_eq_to_matrix(solution_list,[z_0,z_x,z_y]))
            solution_equations.append(solution_list)
#             print(sym.latex(sym.Matrix(solution_list)))

        #This is used to establish boundary conditions
        else:
            zero_vals.append(z_matrix[ycounter][xcounter])
            if xcounter >= element_range:
                q += R.i 
            
            if ycounter >= element_range:
                q += R.j  
         
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
 
 
            for function in interpolation_functions:
                w = function * (v.dot(q)) - sym.Integral(sym.Integral( v.dot(delop(function)),(x,x0,x1)),(y,y0,y1)) - sym.Integral(sym.Integral( f * function,(x,x0,x1)),(y,y0,y1))
                weighted_averages.append(w)
        
            phi_diff_0 = weighted_averages[0] #Minimum at the point (x_0,y_0)
            phi_diff_x = weighted_averages[1] #Minimum at the point (x_1,y_0)
            phi_diff_y = weighted_averages[2] #Minimum at the point (x_0,y_1)
    


            phi_subs_0 = phi_diff_0.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_x = phi_diff_x.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)
            phi_subs_y = phi_diff_y.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y)

            solution_list.append(phi_subs_0.doit())
            solution_list.append(phi_subs_x.doit())
            solution_list.append(phi_subs_y.doit())
            var_list.append([z_0,z_x,z_y])

            solution_matrices.append(sym.linear_eq_to_matrix(solution_list,[z_0,z_x,z_y]))
            solution_equations.append(solution_list)
#             print(sym.latex(sym.Matrix(solution_list)))


#4.  Assemble the element equations into a global matrix. 
#This sets of a global system of equations and the list of unknown variables to be solved
z_vars = [z_matrix[countery][counterx] for countery  in range(len(z_matrix)) for counterx in range(len(z_matrix[0]))]
equation_system = [0 for countery  in range(len(z_matrix)) for counterx in range(len(z_matrix[0]))]

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

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
