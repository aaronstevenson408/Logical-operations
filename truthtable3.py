import itertools
from sympy import symbols, And, Not, Or
import sympy as sp
import json 

def generate_truth_table(truth_table_dict, output_file):
    binary_combinations = list(itertools.product([0, 1], repeat=8))
    print(binary_combinations)
    truth_table = {}
    total_combinations = len(binary_combinations)

    for i, binary_combo in enumerate(binary_combinations, start=1):
        truth_table[i] = [(a, b, c, y) for (a, b, c), y in zip(truth_table_dict.values(), binary_combo)]

        # Print progress for generating truth tables
        print(f'Generating truth table for binary combination {i}/{total_combinations}', end='\r')

    # Save the truth_table as a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(truth_table, json_file, indent=4)

def generate_3_input_logic_gates_with_variables():
    variables = ['a', 'b', 'c']
    equations = []

    for gate_type in ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']:
        for i in range(2**3):  # Loop over all 8 possible input combinations
            equation = ''
            input_values = []
            for j in range(3):
                if (i & (1 << j)) == 0:
                    input_values.append(f'{variables[j]}')
                else:
                    input_values.append(f'(1 - {variables[j]})')
            
            if gate_type == 'AND':
                equation = ' * '.join(input_values)
            elif gate_type == 'OR':
                equation = ' + '.join(input_values)
            elif gate_type == 'NAND':
                equation = f'1 - ({ " * ".join(input_values) })'
            elif gate_type == 'NOR':
                equation = f'1 - ({ " + ".join(input_values) })'
            elif gate_type == 'XOR':
                equation = f'({input_values[0]}) * ({input_values[1]}) * ({input_values[2]}) + ({input_values[0]}) * ({input_values[1]}) * (1 - {input_values[2]}) + ({input_values[0]}) * (1 - {input_values[1]}) * ({input_values[2]}) + (1 - {input_values[0]}) * (1 - {input_values[1]}) * ({input_values[2]})'
            elif gate_type == 'XNOR':
                equation = f'({input_values[0]}) * ({input_values[1]}) * ({input_values[2]}) + ({input_values[0]}) * ({input_values[1]}) * (1 - {input_values[2]}) + ({input_values[0]}) * (1 - {input_values[1]}) * ({input_values[2]}) + (1 - {input_values[0]}) * (1 - {input_values[1]}) * ({input_values[2]})'
            equations.append(equation)
            
            # equations = [sp.sympify(eq) for eq in equations]
    return equations

def match_truth_tables_to_equations(truth_tables, equations):
    matched_tables = {}

    for key, table in truth_tables.items():
        for equation in equations:
            matched = True
            for inputs in table:
                a, b, c, output = inputs
                equation_result = eval(equation)
                matched = equation_result == output

                print(f"Equation: {equation}")
                print(f"Matched: {matched}")
                print(f"Equation Result: {equation_result}")
                print(f"Output: {output}")
                print("\n")

                if not matched:
                    break

            if matched:
                matched_tables[key] = equation
                break

    return matched_tables

def save_truth_table_with_equations_to_file(truth_table, equations, filename):
    with open(filename, 'w') as file:
        for index, rows in truth_table.items():
            file.write(f'Truth Table and Equations for Binary Combination {index}:\n')
            for row, eq in zip(rows, equations[index]):
                file.write(f'{row} --> Equation: {eq}\n')
            file.write('\n')

# Example usage
truth_table_dict = {
    0: (0, 0, 0),
    1: (0, 0, 1),
    2: (0, 1, 0),
    3: (0, 1, 1),
    4: (1, 0, 0),
    5: (1, 0, 1),
    6: (1, 1, 0),
    7: (1, 1, 1)
}

# truth_table = generate_truth_table(truth_table_dict)
# print(truth_table)
# equation_strings  = generate_3_input_logic_gates_with_variables()

# equations = [sp.sympify(eq) for eq in equation_strings]
# print(equations)
# matches =  match_truth_tables_to_equations(truth_table,equations)
# print (matches)

truth_table_dict = {"input1": [0, 1, 0], "input2": [1, 0, 1]}
output_file = "truth_table.json"
generate_truth_table(truth_table_dict, output_file)