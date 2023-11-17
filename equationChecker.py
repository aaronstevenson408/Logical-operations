import sympy as sp
import numpy as np
import numpy as np
import sympy as sp

import numpy as np
import sympy as sp

import numpy as np
import sympy as sp

def evaluate_expression(expression, variable_range, increment, output_bounds):
    # Parse the mathematical expression
    parsed_expression = sp.sympify(expression)
    
    # Extract variable names from the expression
    variable_names = list(parsed_expression.free_symbols)

    # Check if there are any variables in the expression
    if not variable_names:
        return False, []

    # Generate grids of variable values within the specified range
    num_points = int((variable_range[1] - variable_range[0]) / increment) + 1
    grids = [np.linspace(variable_range[0], variable_range[1], num_points) for _ in variable_names]
    grid_combinations = np.array(np.meshgrid(*grids)).T.reshape(-1, len(variable_names))

    # Initialize pass_fail to True, assuming the expression passes
    pass_fail = True
    evaluations = []  # Initialize an empty list to store evaluations

    for values in grid_combinations:
        variable_values = dict(zip(variable_names, values))
        try:
            result = parsed_expression.evalf(subs=variable_values)  # Evaluate the expression with substitutions
        except (ZeroDivisionError, ValueError, TypeError):
            result = None
            pass_fail = False  # Set pass_fail to False for division by zero, non-real, or other errors

        if result is not None:
            # Check for non-real (complex or infinite) results
            if result.is_real and not result.is_infinite:
                is_within_bounds = (output_bounds[0] <= result <= output_bounds[1])
                if not is_within_bounds:
                    pass_fail = False  # Set pass_fail to False if any combination is outside of bounds
            else:
                pass_fail = False  # Set pass_fail to False for non-real or infinite results

        evaluations.append((variable_values, result, is_within_bounds if (result is not None and result.is_real and not result.is_infinite) else False))

    return pass_fail, evaluations



def evaluate_expression_list(expressions, variable_range, increment, output_bounds):
    results_list = []  # Initialize an empty list to store results for each expression

    for expr in expressions:
        pass_fail, evaluations = evaluate_expression(expr, variable_range, increment, output_bounds)
        results_list.append((expr, pass_fail, evaluations))

    return results_list

def print_results(results_list, print_format="short"):
    for expr, pass_fail, evaluations in results_list:
        try:
            has_none = False
            has_infinite = False
            has_non_real = False

            for variables, result, is_within_bounds in evaluations:
                if result is None:
                    has_none = True
                elif not result.is_real:
                    has_non_real = True
                elif result.is_infinite:
                    has_infinite = True

            if print_format == "short":
                summary = f"Expression: {expr}, Pass: {pass_fail}"
                if has_none:
                    summary += ", Result: N/A (None)"
                elif has_non_real:
                    summary += ", Result: N/A (Non-real)"
                elif has_infinite:
                    summary += ", Result: N/A (Infinite)"
                else: 
                    summary += f", Result: {result:.6f}"
                print(summary)

            elif print_format == "long":
                print(f"Expression: {expr}")
                for variables, result, is_within_bounds in evaluations:
                    if result is not None:
                        if result.is_real:
                            if result.is_infinite:
                                print(f"  Variables: {variables}, Result: {result:.6f}, Within Bounds: {is_within_bounds}, Infinite: Yes")
                            else:
                                print(f"  Variables: {variables}, Result: {result:.6f}, Within Bounds: {is_within_bounds}, Infinite: No")
                        else:
                            print(f"  Variables: {variables}, Result: N/A (Non-real), Within Bounds: N/A (Non-real), Infinite: N/A")
                    else:
                        print(f"  Variables: {variables}, Result: N/A, Within Bounds: N/A (Non-real or Infinite), Infinite: N/A")
                print(f"  Overall Pass/Fail: {pass_fail}")

            elif print_format == "pass_fail":
                print(f"Expression: {expr}, Pass: {pass_fail}")

        except Exception as e:
            print(f"Error processing expression: {expr}. Exception: {str(e)}")



# Example usage
if __name__ == "__main__":
    expressions = [
        # "1 - abs(a - b)",
        # "a + (b - 1)",
        # "(a * b) / (a + b)",
        # "abs(a - b) + 1",
        # "(a + 1) - abs(b - 1)",
        # "abs(a - 1) + abs(b - 1)",
        # "abs(a - b + 1)",
        # "a - b + (1 - 1)",
        # "abs((a - 1) - (b - 1))",
        # "a / a + b - b",
        # "abs(a - b) + abs(1 - 1)",
        # "(a * 1) + (b - b)",
        # "(a + 1) * (1 / b)",
        # "abs(a) - abs(b) + 1",
        # "(a + 1) / (b + 1)",
        # "(1 / a) + (1 / b)",
        # "(a - 1) / (b - 1)",
        # "a + (b * 0) + 1",
        # "abs(a) + abs(b) - 1",
        # "(a * 1) - (b * 0)",
        # "a / (b + 1) + (1 - 1)",
        # "(a * 0) + (1 * b) + 1",
        # "abs(a - b) + abs(1 - 0)",
        # "(a + 0) - (b + 0) + 1",
        # "abs((a - 1) - (b + 1))",
        # "a / a + (b - b)",
        # "abs(a) - abs(b - 1)",
        # "(a + 1) - abs(b) + (0 * 0)",
        # "a + (b - 1) + (0 / 0)",
        # "(a * b) / (a + b + 0)",
        # "abs((a + 0) - (b - 0)) + 1",
        # "(a - 1) / (b + 1) + 1",
        # "a + (1 - b) - 0",
        # "abs(a - b) - abs(1 - 0)",
        # "(a + 0) * (0 / b) + 1",
        # "(a + 1) * (1 / b + 0)",
        # "(1 / a) + (0 / b) + 1",
        # "(a - 1) / (0 + b) + 1",
        # "(a + 0) - abs(b) + 1",
        # "abs((a + 1) - (b + 1)) + (0 - 0)",
        # "a / a + (1 - b)",
        # "abs(a) - abs(b) + (1 - 1)",
        #  TODO : still having trouble finding all the divide by zero errors

        # Error processing expression: (a + 1) * (0 / b) - 0. Exception: cannot access local variable 'result' where it is not associated with a value
        # Error processing expression: (a - 1) * (0 / b) + (0 + 1). Exception: cannot access local variable 'result' where it is not associated with a value
        "(a + 1) * (0 / b) - 0",
        "(a - 1) * (0 / b) + (0 + 1)"
    ]

    variable_range = (0, 1)
    increment = 0.5
    output_bounds = (0, 1)

    results_list = evaluate_expression_list(expressions, variable_range, increment, output_bounds)

    print_results(results_list, print_format= "short")