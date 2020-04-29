'''
Created on Feb 13, 2020

@author: Andy Block
'''

import numpy as np
import sympy as sym
# Step 1 - Discretize the Structure

global_mapping = [[2,3],[2,1],[1,3]]
u1,v1,u2,v2,u3,v3 = sym.symbols('u1 v1 u2 v2 u3 v3')
displacement_mapping = [[u1,v1],[u2,v2],[u3,v3]]

Fu1,Fv1,Fu2,Fv2,Fu3,Fv3 = sym.symbols('F_u1 F_v1 F_u2 F_v2 F_u3 F_v3')
force_mapping = [[Fu1,Fv1],[Fu2,Fv2],[Fu3,Fv3]]

global_force_position = [Fu1,Fv1,Fu2,Fv2,Fu3,Fv3]
global_displacement_position = [u1,v1,u2,v2,u3,v3]

# Step 2 - Find all the element properties
area = [32.3 * 10 ** -4,38.7 * 10 ** -4,25.8 * 10 ** -4]

intersection_angle = [90 * np.pi/180,0,135 * np.pi/180]
element_length = [2.54,2.54,2.54 * np.sqrt(2)]
modulus = [6.9 * 10**10, 20.7 * 10**10, 20.7 * 10**10]

elemental_stiffness_matrix = np.matrix([[1,-1],[-1,1]])

indiv_stiffness_matrices =[]

global_force_vector = [u1,v1,u2,v2,u3,v3]

global_stiffness_matrix = np.zeros((6,6))
k = []

uv = []
Fuv = []

#Step 3 - Assemble the Element Matrix
for counter in range(len(global_mapping)):
    
    rotation_matrix = np.matrix([[np.cos(intersection_angle[counter]),0],[np.sin(intersection_angle[counter]),0],[0,np.cos(intersection_angle[counter])],[0,np.sin(intersection_angle[counter])]])
    k.append( (modulus[counter] * area[counter]/element_length[counter]) *  rotation_matrix * elemental_stiffness_matrix*np.transpose(rotation_matrix))

    zoop = []
    zorp = []
    for counter2 in global_mapping[counter]:
        zoop.append(displacement_mapping[counter2 - 1][0])
        zoop.append(displacement_mapping[counter2 - 1][1])
        zorp.append(force_mapping[counter2 - 1][0])
        zorp.append(force_mapping[counter2 - 1][1])
    uv.append(zoop)
    Fuv.append(zorp)
k = [sym.Matrix(np.round(k[counter])) for counter in range(len(k))]
uv = [sym.Matrix(uv[counter]) for counter in range(len(uv))]

#Step 4 - Assemble the Equations for the whole system
matrix_product = [k[counter] * uv[counter] for counter in range(len(k))]
equation_system = [0 for x in range(6)]

for element_counter in range(len(matrix_product)):
    for counter in range(len(matrix_product[element_counter])):
        equation_system[global_force_position.index(Fuv[element_counter][counter])] += matrix_product[element_counter][counter]

equation_system = [sym.Eq(equation_system[counter],global_force_position[counter]) for counter in range(len(equation_system))]

matrix_solution = sym.linear_eq_to_matrix(equation_system,global_displacement_position)

# print(sym.latex(sym.Matrix(equation_system)))
# print(sym.latex(uv))

# print(sym.latex(Fuv))

# print(sym.latex(matrix_product))

print(sym.latex(sym.Matrix(equation_system)))

print(sym.latex(matrix_solution))

#Step 5 - Establish Boundary Conditions 

displacement_boundaries = sym.Matrix([u1,v1,0,0,0,0])
force_boundaries = sym.Matrix([0.0222, -0.111, Fu2,Fv2,Fu3,Fv3])

#Step 6 - Generate the Equations and solve them

force_boundary_matrix = matrix_solution[0] * displacement_boundaries
displacement_equations = [sym.Eq((force_boundary_matrix)[counter],force_boundaries[counter]) for counter in range(len(displacement_boundaries))]

print(sym.latex(force_boundary_matrix))
print(sym.latex(sym.Matrix(displacement_equations)))
solved_variables = sym.linsolve(displacement_equations,[u1,v1,Fu2,Fv2,Fu3,Fv3])

#Step 7 - computer the Stresses and strains in each of the elements. 
final_displacement = [solved_variables.args[0][0].evalf(),solved_variables.args[0][1].evalf(),0,0,0,0]

# print(intersection_angle[0])
element_strain = []
for counter in range(len(uv)):
    displacement_vector = sym.Matrix([final_displacement[global_force_vector.index(uv_objert)] for uv_objert in uv[counter]])
    angle_matrix = sym.Matrix([[sym.cos(intersection_angle[counter]),sym.sin(intersection_angle[counter]),0,0],
                    [0,0,sym.cos(intersection_angle[counter]),sym.sin(intersection_angle[counter])]])
    element_strain.append(angle_matrix * displacement_vector)
#     print(displacement_vector)

# print(solved_variables.args[0][0].evalf())
# print(sym.latex(solved_variables))
# print(sym.latex(element_strain))
# print(sym.latex(displacement_equations))


