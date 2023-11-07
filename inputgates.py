from itertools import product

def generate_truth_tables_3_input_logic_gates():
    input_values = [0, 1]
    input_combinations = list(product(input_values, repeat=3))
    
    truth_tables = []
    
    for gate in ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]:
        truth_table = []
        
        for combination in input_combinations:
            if gate == "AND":
                output = int(all(combination))
            elif gate == "OR":
                output = int(any(combination))
            elif gate == "XOR":
                output = int(combination.count(1) % 2 == 1)
            elif gate == "NAND":
                output = int(not all(combination))
            elif gate == "NOR":
                output = int(not any(combination))
            elif gate == "XNOR":
                output = int(combination.count(1) % 2 == 0)
            
            truth_table.append((combination, output))
        
        truth_tables.append((gate, truth_table))
    
    return truth_tables

if __name__ == "__main__":
    truth_tables = generate_truth_tables_3_input_logic_gates()
    
    for gate, truth_table in truth_tables:
        print(f"Truth table for {gate} gate:")
        for combination, output in truth_table:
            print(f"{combination} -> {output}")
        print()
