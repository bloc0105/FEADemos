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
y_range = num.linspace(low_boundary,high_boundary,number_of_divisions)


elements_with_coordinates =  [ [[x_range[xcounter],x_range[xcounter + 1],y_range[ycounter],y_range[ycounter + 1]] for xcounter in range(number_of_divisions - 1)] for ycounter in range(number_of_divisions - 1)]
intercept_matrix = [[sym.symbols('u_'+str(xcounter)+'_'+str(ycounter)) for xcounter in range(number_of_divisions - 1)] for ycounter in range(number_of_divisions - 1)]

solution_matrix = [[0 for xcounter in range(number_of_divisions - 1) ] for ycounter in range(number_of_divisions - 1)]

x,y, f, u, a, g = sym.symbols('x y f u a g')
original_function = (1 - x) * (1 - y)
f = -1

array1 = [x**(n) for n in range(equ_array_length)]
array2 = [y**(n) for n in range(equ_array_length)]
trial_coefficients = [sym.symbols('a_' + str(counter)) for counter in range(equ_array_length**2)]

trial_functions = [j * k * original_function for k in array1 for j in array2] 


for xcounter in range(number_of_divisions - 1): 
    for ycounter in range(number_of_divisions - 1): 
        sum_array1 = []
        fsum_array1 = []
        for functional_counter in range(len(trial_functions)):
            phi = 0
            for summation_counter in range(len(trial_functions)):
                phi = phi + (sym.diff(trial_functions[functional_counter],x) * sym.diff(trial_functions[summation_counter],x) +  sym.diff(trial_functions[functional_counter],y) * sym.diff(trial_functions[summation_counter],y)) * trial_coefficients[summation_counter]
            sum_array1.append(phi + f * trial_functions[functional_counter])
        functional_array = []
        for counter in range(len(sum_array1)):
            print('x = ' + str(elements_with_coordinates[xcounter][ycounter][0]) + ' y = ' + str(elements_with_coordinates[xcounter][ycounter][2]) + ' Integral - ' + str(counter))
            
            functional_array.append(sym.Eq(2 * sym.N(sym.integrate(sym.integrate(sum_array1[counter],(x,elements_with_coordinates[xcounter][ycounter][0],elements_with_coordinates[xcounter][ycounter][1])),(y,elements_with_coordinates[xcounter][ycounter][2],elements_with_coordinates[xcounter][ycounter][3])),7),0))

        result_set = sym.linsolve(functional_array, trial_coefficients)


        u = 0
        for d in range(len(result_set.args[0])):
            u = u + result_set.args[0][d] * trial_functions[d] 
        solution_matrix[xcounter][ycounter] += u + intercept_matrix[xcounter][ycounter]

X_Grid = []
Y_Grid = []
values = []
x_plot = []
vals_plot = []

element_continuity_equations = []
element_continuity_variables = []
for xcounter in range(len(elements_with_coordinates)): 
    for ycounter in range(len(elements_with_coordinates[0])):
        
        

        current_corner_equation = 0
        previous_corner_equation = 0
        if (xcounter != len(elements_with_coordinates) - 1 and ycounter != len(elements_with_coordinates) - 1):

            previous_corner_equation = solution_matrix[xcounter + 1][ycounter + 1].subs(x,elements_with_coordinates[xcounter + 1][ycounter+ 1][0]).subs(y,elements_with_coordinates[xcounter + 1][ycounter + 1][2])
            
        current_corner_equation = solution_matrix[xcounter][ycounter].subs(x,elements_with_coordinates[xcounter][ycounter][1]).subs(y,elements_with_coordinates[xcounter][ycounter][3])
        element_continuity_equations.append(sym.Eq(current_corner_equation,previous_corner_equation))
        element_continuity_variables.append(intercept_matrix[xcounter][ycounter])
        
element_continuities = sym.linsolve(element_continuity_equations,element_continuity_variables)

print (sym.latex(sym.Matrix(element_continuities.args[0])))

element_equations = [[0 for xcounter in range(len(elements_with_coordinates))]for ycounter in range(len(elements_with_coordinates[0]))]

for xcounter in range(len(elements_with_coordinates)): 
    for ycounter in range(len(elements_with_coordinates[0])):
        continuity_index = element_continuity_variables.index(intercept_matrix[xcounter][ycounter])
        g = sym.lambdify([x,y],solution_matrix[xcounter][ycounter].subs(element_continuity_variables[continuity_index],element_continuities.args[0][continuity_index]),"numpy")
        element_equations[xcounter][ycounter] = g

XG,YG = num.meshgrid(x_range,y_range)
values = num.zeros([len(x_range), len(y_range)])
for x_count in range(len(XG)):
    for y_count in range(len(XG[0])):
        for element_x_counter in range(len(elements_with_coordinates)):
            for element_y_counter in range(len(elements_with_coordinates[0])):
                if (XG[y_count][x_count] >= elements_with_coordinates[element_x_counter][element_y_counter][0] or XG[y_count][x_count] <= elements_with_coordinates[element_x_counter][element_y_counter][1]) and (YG[y_count][x_count] >= elements_with_coordinates[element_x_counter][element_y_counter][2] or YG[y_count][x_count] <= elements_with_coordinates[element_x_counter][element_y_counter][3]):
                    values[y_count][x_count] = element_equations[element_x_counter][element_y_counter](XG[y_count][x_count],YG[y_count][x_count])
        
plott.contourf(XG,YG,values,120)
               

plott.colorbar()
plott.show()

    
