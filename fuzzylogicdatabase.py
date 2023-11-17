import sqlite3
import os
import itertools 
from itertools import product

DB_FILE = 'truth_tables.db'

def create_database(db_file=DB_FILE):
    # Create a new SQLite database file
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    # Create TruthTables table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TruthTables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logic_name TEXT NOT NULL,
            expression TEXT NOT NULL,
            image_path TEXT,  -- Store the file path instead of the image itself
            UNIQUE(logic_name)
        )
    ''')

    # Create TruthTableRows table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TruthTableRows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            truth_table_id INTEGER,
            row_number INTEGER,
            input1 INTEGER,
            input2 INTEGER,
            input3 INTEGER,
            output INTEGER,
            FOREIGN KEY (truth_table_id) REFERENCES TruthTables(id) ON DELETE CASCADE
        )
    ''')

    connection.commit()
    connection.close()

def create_truth_table(logic_name, expression, image_path=None, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    try:
        cursor.execute('''
            INSERT INTO TruthTables (logic_name, expression, image_path)
            VALUES (?, ?, ?)
        ''', (logic_name, expression, image_path))
        truth_table_id = cursor.lastrowid
        connection.commit()
    except sqlite3.IntegrityError:
        print(f"Truth table with logic name '{logic_name}' already exists.")
        truth_table_id = None

    connection.close()
    return truth_table_id

def read_truth_table(truth_table_id, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute('''
        SELECT * FROM TruthTables
        WHERE id = ?
    ''', (truth_table_id,))
    result = cursor.fetchone()

    connection.close()
    return result

def update_truth_table(truth_table_id, new_logic_name, new_expression, new_image_path=None, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    try:
        cursor.execute('''
            UPDATE TruthTables
            SET logic_name = ?, expression = ?, image_path = ?
            WHERE id = ?
        ''', (new_logic_name, new_expression, new_image_path, truth_table_id))
        connection.commit()
    except sqlite3.IntegrityError:
        print(f"Cannot update truth table. Logic name '{new_logic_name}' already exists.")

    connection.close()

def delete_truth_table(truth_table_id, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute('''
        DELETE FROM TruthTables
        WHERE id = ?
    ''', (truth_table_id,))
    connection.commit()

    connection.close()

def create_truth_table_row(truth_table_id, row_number, input1, input2, input3, output, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute('''
        INSERT INTO TruthTableRows (truth_table_id, row_number, input1, input2, input3, output)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (truth_table_id, row_number, input1, input2, input3, output))
    row_id = cursor.lastrowid
    connection.commit()

    connection.close()
    return row_id

def read_truth_table_row(truth_table_row_id, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute('''
        SELECT * FROM TruthTableRows
        WHERE id = ?
    ''', (truth_table_row_id,))
    result = cursor.fetchone()

    connection.close()
    return result

def update_truth_table_row(truth_table_row_id, new_row_number, new_input1, new_input2, new_input3, new_output, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute('''
        UPDATE TruthTableRows
        SET row_number = ?, input1 = ?, input2 = ?, input3 = ?, output = ?
        WHERE id = ?
    ''', (new_row_number, new_input1, new_input2, new_input3, new_output, truth_table_row_id))
    connection.commit()

    connection.close()

def delete_truth_table_row(truth_table_row_id, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute('''
        DELETE FROM TruthTableRows
        WHERE id = ?
    ''', (truth_table_row_id,))
    connection.commit()

    connection.close()

def truth_table_inputs(n, values):
    """
    Generates a list of tuples of all possible truth table inputs for n variables.

    Args:
        n: The number of truth table inputs.
        values: A list of the values the truth table inputs could be.

    Returns:
        A list of tuples of all possible truth table inputs.
    """

    # Create a list of empty tuples, one for each truth table input.
    inputs = [[] for _ in range(n)]

    # Recursively generate all possible combinations of truth table inputs.
    def generate_inputs(i):
        if i == n:
            # We've reached the end of the truth table inputs.
            yield tuple(inputs)
        else:
            # Iterate over all possible values for the current truth table input.
            for value in values:
                inputs[i] = value
                yield from generate_inputs(i + 1)

    # Generate all possible combinations of truth table inputs and return them as a list.
    return list(generate_inputs(0))
# Generate a truth table for 2 inputs with possible values True and False.

def generate_permutations(n_inputs, n_outputs, output_values):
    """
    Generates a list of tuples representing all the permutations of outputs for the list of tuples of all possible truth table inputs.

    Args:
        n_inputs: The number of truth table inputs.
        n_outputs: The number of output columns.
        output_values: A list of the values the outputs could be.

    Returns:
        A list of tuples representing all the permutations of outputs for the list of tuples of all possible truth table inputs.
    """
    inputs = truth_table_inputs(n_inputs, output_values)
    return list(itertools.product(output_values, repeat=n_outputs * len(inputs))) 

# Example usage:
n = 3  # Number of truth table inputs
input_values = [True, False]  # Values the truth table inputs could be
input_tuples = truth_table_inputs(n, input_values)

num_output_columns = 1 # Number of output columns
output_values = [True,False]  # Values the outputs could be

output_permutations = generate_permutations(len(input_tuples), num_output_columns, output_values)
print(output_permutations)
# # Example usage:
# create_database()

# # Example usage for TruthTables:
# truth_table_id = create_truth_table("AND Gate", "A AND B", image_path=None)
# retrieved_truth_table = read_truth_table(truth_table_id)
# update_truth_table(truth_table_id, "Updated AND Gate", "A AND B", image_path=None)
# delete_truth_table(truth_table_id)

# # Example usage for TruthTableRows:
# row_id = create_truth_table_row(truth_table_id, 1, 0, 1, 1, 1)
# retrieved_row = read_truth_table_row(row_id)
# update_truth_table_row(row_id, 2, 1, 0, 1, 0)
# delete_truth_table_row(row_id)
