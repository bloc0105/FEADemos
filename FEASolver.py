import numpy as num
import matplotlib as plotting
from matplotlib import pyplot as plott
import sympy as sym

low_boundary = 0 # The lower boundary of the graph
high_boundary = 1 # The upper boundary of the graph
grid_points = 6 # The number of nodes that exist along the axes of the boundary (inclusive)

number_of_divisions = (high_boundary - low_boundary) * grid_points #each quadrant will have the same number of nodes in it

x_range = num.linspace(low_boundary,high_boundary,number_of_divisions)
y_range = num.linspace(low_boundary,high_boundary,number_of_divisions)

element_range = number_of_divisions - 1


XGrid,YGrid = num.meshgrid(x_range, y_range)

z_matrix = [[sym.symbols('z_x' + str(counterx) + '_y' + str(countery)) for counterx in range(len(XGrid))] for countery in range(len(XGrid[0]))]

# print(sym.latex(sym.Matrix(XGrid)))
# print(sym.latex(sym.Matrix(YGrid)))
# print(sym.latex(sym.Matrix(z_matrix)))

#comment!
x,y, f, u = sym.symbols('x y f u')
x0,y0,y1,x1,phi0,phix,phiy = sym.symbols('x_0 y_0 y_1 x_1 phi_0 phi_x phi_y')

m = (phix - phi0)/(x1 - x0)
n = (phiy - phi0)/(y1 - y0)
b = phi0 - m * x0 - n * y0

phi_final = m * x + n * y + b

# print(sym.latex(phi_final))

# print(sym.latex(sym.diff(phi_final,x)))
f = -1
phi = (sym.diff(phi_final,x))**2  +  (sym.diff(phi_final,y))**2 + 2 * f * phi_final

# print(sym.latex(sym.simplify(phi)))

phi_integral = sym.integrate(sym.integrate(phi,(x,x0,x1)),(y,y0,y1))

# print(sym.latex(sym.simplify(phi_integral)))


phi_diff_0 = sym.diff(phi_integral, phi0)
phi_diff_x = sym.diff(phi_integral, phix)
phi_diff_y = sym.diff(phi_integral, phiy)

phi_solves = []

phi_solves.append(sym.simplify(phi_diff_0.subs(x0,XGrid[0][0]).subs(x1,XGrid[0][1]).subs(y0,YGrid[0][0]).subs(y1,YGrid[1][0])))
phi_solves.append(sym.simplify(phi_diff_x.subs(x0,XGrid[0][0]).subs(x1,XGrid[0][1]).subs(y0,YGrid[0][0]).subs(y1,YGrid[1][0])))
phi_solves.append(sym.simplify(phi_diff_y.subs(x0,XGrid[0][0]).subs(x1,XGrid[0][1]).subs(y0,YGrid[0][0]).subs(y1,YGrid[1][0])))

# print(sym.latex(sym.Matrix(phi_solves)))

q = sym.linsolve(phi_solves,[phi0,phix,phiy])

# print(sym.latex(sym.Matrix([phi0,phix,phiy])))

# print(sym.latex(q))

# print (sym.latex(phi_final))
# print (sym.latex(phi))
# print (sym.latex(phi_integral))
# print(sym.latex(phi_diff_y))


var_list = []
solution_matrices = []
solution_equations = []
zero_vals = []
for xcounter in range(number_of_divisions):
    for ycounter in range(number_of_divisions):

        if xcounter < element_range and ycounter < element_range:
            solution_list = []
            x_max = XGrid[ycounter][xcounter + 1]
            x_min = XGrid[ycounter][xcounter]

            y_max = YGrid[ycounter +1][xcounter]
            y_min = YGrid[ycounter][xcounter]

            z_y = z_matrix[ycounter + 1][xcounter]
            z_x = z_matrix[ycounter][xcounter + 1]
            z_0 = z_matrix[ycounter][xcounter]


            phi_subs_0 = sym.N(phi_diff_0.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y),2)
            phi_subs_x = sym.N(phi_diff_x.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y),2)
            phi_subs_y = sym.N(phi_diff_y.subs(x0,x_min).subs(x1,x_max).subs(y0,y_min).subs(y1,y_max).subs(phi0,z_0).subs(phix,z_x).subs(phiy,z_y),2)

            solution_list.append(phi_subs_0)
            solution_list.append(phi_subs_x)
            solution_list.append(phi_subs_y)
            var_list.append([z_0,z_x,z_y])

            solution_matrices.append(sym.linear_eq_to_matrix(solution_list,[z_0,z_x,z_y]))
            solution_equations.append(solution_list)

        else:
            zero_vals.append(z_matrix[ycounter][xcounter])

z_vars = [z_matrix[countery][counterx] for countery  in range(len(z_matrix)) for counterx in range(len(z_matrix[0]))]
equation_system = [0 for countery  in range(len(z_matrix)) for counterx in range(len(z_matrix[0]))]

# print(sym.latex(sym.Matrix(equation_system)))


for solution_counter in range(len(solution_equations)):
    for eq_counter in range(len(solution_equations[solution_counter])):

        equation_system[z_vars.index(var_list[solution_counter][eq_counter])] += solution_equations[solution_counter][eq_counter]


# equation_system.append(z_vars[len(z_vars) - 1])
# print(sym.latex(sym.Matrix(z_vars)))
# print(sym.latex(sym.Matrix(equation_system)))

solution_matrix =  sym.linear_eq_to_matrix(equation_system,z_vars)

eq_matrix = solution_matrix[0]
result_matrix = solution_matrix[1]

# print (sym.latex(solution_matrix))
z_vars_boundary = []

for each_var in z_vars:
    if each_var in zero_vals:
        z_vars_boundary.append(0)
    else:
        z_vars_boundary.append(each_var)
        
# print(sym.latex(sym.Matrix(z_vars_boundary)))

boundary_equations = eq_matrix * sym.Matrix(z_vars_boundary)

# print(sym.latex(sym.Matrix(boundary_equations))
boundary_eq_with_result = []
for counter in range(len(boundary_equations)):
    if z_vars_boundary[counter] != 0:
        
        boundary_eq_with_result.append(sym.Eq(boundary_equations[counter],result_matrix[counter]))
    else:
        boundary_eq_with_result.append(sym.Eq(z_vars[counter],0))
# print(sym.latex(sym.Matrix(boundary_eq_with_result)))    

# print(sym.latex(solution_matrix[0]))
# print(sym.latex(sym.Matrix(zero_vals)))
# print(sym.latex(sym.Matrix(z_var_list)))
# print(sym.latex(sym.Matrix(var_list)))


# print(len(z_var_list))
# print(sym.latex(sym.Matrix(z_var_list)))
# print(len(solution_list))

# print(sym.latex(sym.Matrix(solution_list)))

resultset = sym.linsolve(boundary_eq_with_result,z_vars)
# resultset = sym.linsolve(equation_system,z_vars)
result_vals = [resultset.args[0][counter] for counter in range(len(z_vars))]

result_grid = num.zeros([number_of_divisions,number_of_divisions])

for ycounter in range(len(z_matrix)):
    for xcounter in range(len(z_matrix[0])):
        result_grid[ycounter][xcounter] += result_vals[z_vars.index(z_matrix[ycounter][xcounter])]

# print(sym.latex(resultset))


plott.contourf(XGrid,YGrid,result_grid, 150)
plott.colorbar()
plott.show()
