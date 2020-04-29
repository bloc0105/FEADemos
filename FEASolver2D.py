import numpy as num
import matplotlib as plotting
from matplotlib import pyplot as plott
import sympy as sym

low_boundary = 0 # The lower boundary of the graph
high_boundary = 1 # The upper boundary of the graph
grid_points = 10 # The number of nodes that exist along the axes of the boundary (inclusive)
equ_array_length = 1
sub_grid_size = 5

number_of_divisions = (high_boundary - low_boundary) * grid_points #each quadrant will have the same number of nodes in it 

x_range = num.linspace(low_boundary,high_boundary,number_of_divisions)

y_vars = [sym.symbols('y_' + str(counter)) for counter in range(number_of_divisions)]

# print(number_of_divisions)
# print(len(y_vars))
element_range = number_of_divisions - 1

x,y, f, u = sym.symbols('x y f u')
xstart,ystart,yend,xend= sym.symbols('x_start y_start y_end x_end')

m = (yend - ystart)/(xend - xstart)
b = ystart - m * xstart 

y_final  = m * x + b

# print(sym.latex(m))


# print(sym.latex(sym.diff(phi_final,x)))
f = -1
phi = 1 / 2 * (sym.diff(y_final,x))**2  + f * y_final

# print(sym.latex(sym.simplify(phi)))

phi_integral = sym.integrate(phi,(x,xstart,xend))

# print(sym.latex(sym.simplify(phi_integral)))


phi_diff_0 = sym.diff(phi_integral, ystart)
phi_diff_x = sym.diff(phi_integral, yend)

var_list= []
solution_matrices = []
solution_equations = []

zero_boundaries = []
for xcounter in range(number_of_divisions):
    if xcounter < number_of_divisions - 1:
        solution_list = []
        x_high = x_range[xcounter]
        x_low = x_range[xcounter + 1]
        y_high = y_vars[xcounter]
        y_low = y_vars[xcounter + 1]
        eq0 = phi_diff_0.subs(xstart,x_high).subs(xend,x_low).subs(ystarhttp://www.bodybuilding.com/exercises/finder/lookupt,y_high).subs(yend,y_low)  
        eq1 = phi_diff_x.subs(xstart,x_high).subs(xend,x_low).subs(ystart,y_high).subs(yend,y_low)  
        solution_list.append(eq0)
        solution_list.append(eq1)
        var_list.append([y_high,y_low])
        solution_equations.append(solution_list)
        solution_matrices.append(sym.linear_eq_to_matrix(solution_list,[y_high,y_low]))
    
    else:
        zero_boundaries.append(y_high)
        
        
        
                
equation_system = [0 for derp in range(number_of_divisions)]

# print(sym.latex(solution_matrices))


for solution_counter in range(len(solution_equations)):
    for eq_counter in range(len(solution_equations[solution_counter])):
        
        equation_system[y_vars.index(var_list[solution_counter][eq_counter])] += solution_equations[solution_counter][eq_counter]

# print(sym.latex(sym.Matrix(equation_system))) 
solution_matrix =  sym.linear_eq_to_matrix(equation_system,y_vars)


print(sym.latex(solution_matrix[0]))

y_vars_boundary = []

for each_var in y_vars:
    if each_var in zero_vals:
        zero_boundaries.append(0)
    else:
        zero_boundaries.append(each_var)

solutions_with_result = solution_matrix[0].col_insert(number_of_divisions,solution_matrix[1])

# solutions_with_result =[solution_matrix[0][counter].append(solution_matrix[1][counter]) for counter in range(len(solution_matrix[0]))]

# print(sym.latex(sym.Matrix(solutions_with_result)))
# print(sym.latex(solutions_with_result.rref()))

# print(sym.latex(sym.Matrix(equation_system)))

# equation_system.append(y_vars[0])
# equation_system.append(y_vars[len(y_vars) - 1])
# print(sym.latex(solutio# print(sym.latex(resultset))n_matrix))

zoop = y_vars[1:len(y_vars) - 1]
# print(sym.latex(zoop))
resultset = sym.linsolve(equation_system,y_vars)


# print(sym.latex(sym.Matrix(solution_list)))
# print(sym.latex(solution_matrices))
# print(sym.latex(sym.Matrix(solution_list)))

# print(sym.latex(resultset))
result_matrix = [something for something in  resultset.args[0]]
# print(sym.Matrix(result_matrix))

# plott.plot(x_range,result_matrix)



# plott.colorbar()
# plott.show()

    
