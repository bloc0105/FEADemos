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
grid_points = 4 # The number of nodes that exist along the axes of the boundary (inclusive)

#1. Discretize the system.
number_of_divisions = (high_boundary - low_boundary) * grid_points  #each quadrant will have the same number of nodes in it

x_range = num.linspace(low_boundary,high_boundary,number_of_divisions)
y_range = num.linspace(low_boundary,high_boundary,number_of_divisions)

element_range = number_of_divisions - 1

x_range_reduced = x_range[0:len(x_range) - 1] + num.diff(x_range)/2
x_range_reduced = x_range[0:len(x_range) - 1] + num.diff(x_range)/2

# print(x_range)
# print(x_range_reduced)


XGrid,YGrid = num.meshgrid(x_range, y_range) #Discretize the space into rectangular elements.

X_Grid_shift_1 = num.delete(XGrid, len(XGrid) - 1,1) + num.diff(x_range)[0]/2
Y_Grid_shift_1 = num.delete(YGrid, len(YGrid) - 1,1)

X_Grid_shift_1 = num.transpose(X_Grid_shift_1)
Y_Grid_shift_1 = num.transpose(Y_Grid_shift_1)


X_Grid_shift_2 = num.delete(XGrid, len(XGrid) - 1,0)
Y_Grid_shift_2 = num.delete(YGrid, len(YGrid) - 1,0) + num.diff(y_range)[0]/2


XGrid_expanded = num.concatenate((XGrid,X_Grid_shift_1,X_Grid_shift_2),0)
YGrid_expanded = num.concatenate((YGrid,Y_Grid_shift_1,Y_Grid_shift_2),0)


# print(sym.latex(sym.Matrix(XGrid_expanded)))
# print(sym.latex(sym.Matrix(YGrid_expanded)))

value_pairs = [[XGrid_expanded[rowcounter][columncounter],YGrid_expanded[rowcounter][columncounter]] for rowcounter in range(len(XGrid_expanded)) for columncounter in range(len(XGrid_expanded[0]))]
pairs_x = [XGrid_expanded[rowcounter][columncounter] for rowcounter in range(len(XGrid_expanded)) for columncounter in range(len(XGrid_expanded[0]))]
pairs_y = [YGrid_expanded[rowcounter][columncounter] for rowcounter in range(len(XGrid_expanded)) for columncounter in range(len(XGrid_expanded[0]))]

plott.scatter(pairs_x,pairs_y)

for pair_index, pair in enumerate(value_pairs):
    plott.text(pair[0] + .02, pair[1], str(pair_index))


plott.show()

    
# print(sym.latex(sym.Matrix(value_pairs)))
z_matrix = [sym.symbols('z_' + str(counter)) for counter in range(len(value_pairs))]
                        
# print(sym.latex(sym.Matrix(z_matrix)))

# print(sym.latex(sym.Matrix(XGrid)))


#This labels all the "nodes" in the system.  Each node is a sympy variable that will be solved


#Variables to be used in the calculation.  
x, y, f= sym.symbols('x y f')
x0,y0,y1,x1,phi0,phix,phiy = sym.symbols('x_0 y_0 y_1 x_1 phi_0 phi_x phi_y')
phixx,phiyy = sym.symbols('phi_xx phi_yy')
x2,y2 = sym.symbols('x2 y2')

trial_function = (1 - x**2) * (1 - y**2) 


#2. Set up The Element Parameters

ax,bx,ay,by,c = sym.symbols('a_x b_x a_y b_y c')
#This creates the template element by approximating it as a plane.  
eq1 = sym.Eq(phi0, ax * x0**2 + bx * x0 + ay * y0**2 + by * y0 + c)
eq2 = sym.Eq(phix, ax * x1**2 + bx * x1 + ay * y0**2 + by * y0 + c)
eq3 = sym.Eq(phixx, ax * x2**2 + bx * x2 + ay * y0**2 + by * y0 + c)
eq4 = sym.Eq(phiy, ax * x0**2 + bx * x0 + ay * y1**2 + by * y1 + c)
eq5 = sym.Eq(phiyy, ax * x0**2 + bx * x0 + ay * y2**2 + by * y2 + c)
result = sym.nonlinsolve([eq1,eq2,eq3,eq4,eq5],[ax,ay,bx,by,c])

# print(sym.latex(result))
phi_final = result.args[0][0] * x**2 + result.args[0][1] * y**2 + result.args[0][2] * x  + result.args[0][3] * y + result.args[0][4]  

# print(sym.latex(phi_final))
f = -1  # This is the equation being solved

phi_solved = sym.diff(phi_final,x,x) + sym.diff(phi_final,y,y)
residual = phi_solved - f
# print(sym.latex(phi_solved))


weighted_average = sym.integrate(sym.integrate(residual * trial_function,(x,x0,x2)),(y,y0,y2))
weighted_average_x2 = sym.integrate(sym.integrate(residual * trial_function * x**2,(x,x0,x2)),(y,y0,y2))
weighted_average_x4 = sym.integrate(sym.integrate(residual * trial_function * x**4,(x,x0,x2)),(y,y0,y2))
weighted_average_y2 = sym.integrate(sym.integrate(residual * trial_function * y**2,(x,x0,x2)),(y,y0,y2))
weighted_average_y4 = sym.integrate(sym.integrate(residual * trial_function * y**4,(x,x0,x2)),(y,y0,y2))

# print(sym.latex(weighted_average))

#3. Compute the Element Matrices. 
var_list = []
solution_matrices = []
solution_equations = []


#Each set of three points forms a right triangle. The triangles are solved and made into a matrix. 
for xcounter in range(number_of_divisions):
    for ycounter in range(number_of_divisions):

            
        phi0_element = 0
        phix_element = 0
        phixx_element = 0
        phiy_element = 0
        phiyy_element = 0

        if xcounter < element_range and ycounter < element_range:
            solution_list = []
         
            x_max = XGrid[ycounter][xcounter + 1]
            x_min = XGrid[ycounter][xcounter]

            y_max = YGrid[ycounter +1][xcounter]
            y_min = YGrid[ycounter][xcounter]

            for pair_index, pair in enumerate(value_pairs):
                if pair[0] == x_min and pair[1] == y_min:
                    phi0_element = z_matrix[pair_index]
                elif pair[0] == x_min and pair[1] == y_max:
                    phiyy_element = z_matrix[pair_index]
                elif pair[0] == x_max and pair[1] == y_min:
                    phixx_element = z_matrix[pair_index]
                elif pair[0] == x_min and pair[1] > y_min and pair[1] < y_max:
                    phiy_element = z_matrix[pair_index]
                elif pair[0] > x_min and pair[0] < x_max and pair[1] == y_min:
                    phix_element = z_matrix[pair_index]


            weighted_average_subs = weighted_average.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_subs = weighted_average_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_subs = weighted_average_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_subs = weighted_average_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
#             if xcounter == 0 and ycounter == 0:
#                 print(sym.latex(weighted_average_subs))

            weighted_average_x2_subs = weighted_average_x2.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_x2_subs = weighted_average_x2_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_x2_subs = weighted_average_x2_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_x2_subs = weighted_average_x2_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
            
            weighted_average_x4_subs = weighted_average_x4.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_x4_subs = weighted_average_x4_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_x4_subs = weighted_average_x4_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_x4_subs = weighted_average_x4_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
            
            weighted_average_y2_subs = weighted_average_y2.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_y2_subs = weighted_average_y2_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_y2_subs = weighted_average_y2_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_y2_subs = weighted_average_y2_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)

            weighted_average_y4_subs = weighted_average_y4.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_y4_subs = weighted_average_y4_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_y4_subs = weighted_average_y4_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_y4_subs = weighted_average_y4_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
            
    
            solution_list.append(weighted_average_subs)
            solution_list.append(weighted_average_x2_subs)
            solution_list.append(weighted_average_x4_subs)
            solution_list.append(weighted_average_y2_subs)
            solution_list.append(weighted_average_y4_subs)
            
            var_list.append([phi0_element,phix_element,phiy_element,phiyy_element,phixx_element])
            solution_equations.append(solution_list)

         
        #If not on a lower boundary, make a right triangle in the -x -y direction. 
        if xcounter > 0 and ycounter > 0 : 
            solution_list = []
            x_max = XGrid[ycounter][xcounter - 1]
            x_min = XGrid[ycounter][xcounter]
 
            y_max = YGrid[ycounter - 1][xcounter]
            y_min = YGrid[ycounter][xcounter]
 
            for pair_index, pair in enumerate(value_pairs):
                if pair[0] == x_min and pair[1] == y_min:
                    phi0_element = z_matrix[pair_index]
                elif pair[0] == x_min and pair[1] == y_max:
                    phiyy_element = z_matrix[pair_index]
                elif pair[0] == x_max and pair[1] == y_min:
                    phixx_element = z_matrix[pair_index]
                elif pair[0] == x_min and pair[1] < y_min and pair[1] > y_max:
                    phiy_element = z_matrix[pair_index]
                elif pair[0] < x_min and pair[0] > x_max and pair[1] == y_min:
                    phix_element = z_matrix[pair_index]



            weighted_average_subs = weighted_average.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_subs = weighted_average_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_subs = weighted_average_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_subs = weighted_average_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)

            weighted_average_x2_subs = weighted_average_x2.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_x2_subs = weighted_average_x2_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_x2_subs = weighted_average_x2_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_x2_subs = weighted_average_x2_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
            
            weighted_average_x4_subs = weighted_average_x4.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_x4_subs = weighted_average_x4_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_x4_subs = weighted_average_x4_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_x4_subs = weighted_average_x4_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
            
            weighted_average_y2_subs = weighted_average_y2.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_y2_subs = weighted_average_y2_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_y2_subs = weighted_average_y2_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_y2_subs = weighted_average_y2_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)

            weighted_average_y4_subs = weighted_average_y4.subs(x0,x_min).subs(x2,x_max).subs(y0,y_min).subs(y2,y_max)
            weighted_average_y4_subs = weighted_average_y4_subs.subs(x1, num.average([x_min,x_max])).subs(y1, num.average([y_min,y_max]))
            weighted_average_y4_subs = weighted_average_y4_subs.subs(phi0,phi0_element).subs(phix,phix_element).subs(phiy, phiy_element)
            weighted_average_y4_subs = weighted_average_y4_subs.subs(phixx,phixx_element).subs(phiyy,phiyy_element)
            
                        
            solution_list.append(weighted_average_subs)
            solution_list.append(weighted_average_x2_subs)
            solution_list.append(weighted_average_x4_subs)
            solution_list.append(weighted_average_y2_subs)
            solution_list.append(weighted_average_y4_subs)
            
            var_list.append([phi0_element,phix_element,phiy_element,phiyy_element,phixx_element])
            solution_equations.append(solution_list)


#4.  Assemble the element equations into a global matrix. 
#This sets of a global system of equations and the list of unknown variables to be solved
z_vars = z_matrix
# print(sym.latex(sym.Matrix(z_vars)))
# print(sym.latex(sym.Matrix(var_list)))
# print("============================================================================================")
equation_system = [0 for counter  in z_matrix]

#This loop assembles each of the individual element equations into the global list of equations.
for solution_counter in range(len(solution_equations)):
#     print(sym.latex(sym.Matrix(solution_equations[solution_counter])))
#     print("------------------------------------------------------------------------------------------")
    for eq_counter in range(len(solution_equations[solution_counter])):
#         print(var_list[solution_counter][eq_counter])
        derp =  equation_system[z_vars.index(var_list[solution_counter][eq_counter])]

        equation_system[z_vars.index(var_list[solution_counter][eq_counter])] += solution_equations[solution_counter][eq_counter]

#Make the equations into a matrix to make them easier to solve. 
solution_matrix =  sym.linear_eq_to_matrix(equation_system,z_vars)

eq_matrix = solution_matrix[0]
print(sym.latex(eq_matrix))
# print(sym.latex(solution_matrix))
result_matrix = solution_matrix[1]

zero_vals = []
for pair_index, pair in enumerate(value_pairs):
    if pair[0] == 1 or pair[1] == 1:
        zero_vals.append(z_vars[pair_index])

# print(sym.latex(sym.Matrix(zero_vals)))
z_vars_boundary = []

#5. Impose Boundary conditions. Zero at x = 1 and y = 1. This replaces all of the nodes in those positions with zeros.
for each_var in z_vars:
    if each_var in zero_vals:
        z_vars_boundary.append(0)
    else:
        z_vars_boundary.append(each_var)
 
#This applies the boundary conditions to the system and converts it back into a system of equations.        
boundary_equations = eq_matrix * sym.Matrix(z_vars_boundary)

#Conditions where the equation already solves to zero are a problem, because 0 = 0 will not evaluate
#This replaces the boundary condition and sets a specific variable to 0 so it can be solved. 
boundary_eq_with_result = []
for counter in range(len(boundary_equations)):
    if z_vars_boundary[counter] != 0:
         
        boundary_eq_with_result.append(sym.Eq(boundary_equations[counter],result_matrix[counter]))
    else:
        boundary_eq_with_result.append(sym.Eq(z_vars[counter],0))

#6. Solve the System. Sympy linsolve makes short work of that.   
resultset = sym.linsolve(boundary_eq_with_result,z_vars)
print(resultset)

#7 Use the computed results to determine desired results. 
#In most FEA solutions, this would be stresses or fluid flow, but in this case, it's just the Z-Values.
result_vals = [resultset.args[0][counter] for counter in range(len(z_vars))]

result_grid = num.zeros([len(XGrid_expanded),len(XGrid_expanded[0])])

#Make a meshgrid with the Z-Values in it
# varcount = 0
for zcounter in range(len(z_matrix)):
    for ycounter in range(len(XGrid_expanded)):
        for xcounter in range(len(XGrid_expanded[0])):
            if value_pairs[zcounter][0] == XGrid_expanded[ycounter][xcounter] and value_pairs[zcounter][1] == YGrid_expanded[ycounter][xcounter]:
                
                result_grid[ycounter][xcounter] += result_vals[zcounter]
#         varcount += 1

#Plot the results
plott.contourf(XGrid_expanded,YGrid_expanded,result_grid, 150)
plott.colorbar()
plott.show()
