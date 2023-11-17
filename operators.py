#TODO visualization of the different operations , (truth tables, line graphs (for unary), and heat maps(for binary)) kinda done needs a bit of work

# What should be the scale(keep track of if true)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from prettytable import PrettyTable
import io
from PIL import Image
import os

class TruthScale:
    RANGES = {
        (0.000, 0.001): "False",
        (0.001, 0.011): "Mostly False",
        (0.011, 0.021): "Moderately False",
        (0.021, 0.031): "Somewhat False",
        (0.031, 0.041): "Tending to False",
        (0.041, 0.061): "Neutral",
        (0.061, 0.071): "Tending to True",
        (0.071, 0.081): "Somewhat True",
        (0.081, 0.091): "Moderately True",
        (0.091, 0.099): "Mostly True",
        (0.099, 0.100): "True"
    }

    def __init__(self, value):
        self.value = value
        self.label = self.get_label()

    def get_label(self):
        for (min_range, max_range), label in self.RANGES.items():
            if min_range <= self.value <= max_range:
                return label
        return "Outside Range"  # Default label for values outside the defined ranges

    @staticmethod
    def get_gradient_color(value_or_truthscale_str):
        # Define the color map for the gradient (e.g., red to white to green)
        gradient = LinearSegmentedColormap.from_list("custom_gradient", ["red", "white", "green"], N=256)

        if isinstance(value_or_truthscale_str, (int, float)):
            # If a decimal value is provided
            value = value_or_truthscale_str
        elif isinstance (value_or_truthscale_str, str):
            # If a TruthScale string is provided, convert it to a decimal
            value = TruthScale.truthscale_to_decimal(value_or_truthscale_str)
            if value is None:
                return None
        else:
            # Invalid input
            return None

        norm_value = (value - 0.0) / (1.0 - 0.0)  # Normalize value to the range [0, 1]
        color = gradient(norm_value)
        return color

    @staticmethod
    def truthscale_to_decimal(truthscale_str):
        for (min_range, max_range), label in TruthScale.RANGES.items():
            if label == truthscale_str:
                return round((min_range + max_range) / 2, 2)  # Round to 2 decimal places
        return None

    @staticmethod
    def decimal_to_truthscale(decimal_value):
        decimal_value = round(decimal_value, 2)  # Round the value to 2 decimal places
        for (min_range, max_range), label in TruthScale.RANGES.items():
            if min_range <= decimal_value <= max_range:
                return label
        return "Outside Range"  # Default label for values outside the defined ranges

class FunctionVisualizer:
    def __init__(self, functions):
        self.functions = functions

    def truth_table(self, input_choices='boolean'):
        for function in self.functions:
            table = PrettyTable()
            if function.__code__.co_argcount == 1:
                table.field_names = ['Input', 'Output']
                input_values = []
                if input_choices == 'boolean':
                    input_values = [True, False]
                elif input_choices == 'inc0.1':
                    input_values = [i / 10.0 for i in range(11)]
                elif input_choices == 'inc0.25':
                    input_values = [i / 4.0 for i in range(5)]
                elif input_choices == 'inc0.5':
                    input_values = [1, 0.5, 0]
                for value in input_values:
                    table.add_row([value, function(value)])
            elif function.__code__.co_argcount == 2:
                table.field_names = ['Input A', 'Input B', 'Output']
                input_values = []
                if input_choices == 'boolean':
                    input_values = [(True, True), (True, False), (False, True), (False, False)]
                elif input_choices == 'inc0.1':
                    input_values = [(i / 10.0, j / 10.0) for i in range(11) for j in range(11)]
                elif input_choices == 'inc0.25':
                    input_values = [(i / 4.0, j / 4.0) for i in range(5) for j in range(5)]
                elif input_choices == 'inc0.5':
                    input_values = [(1, 1), (1, 0.5), (1, 0), (0.5, 1), (0.5, 0.5), (0.5, 0), (0, 1), (0, 0.5), (0, 0)]
                for a, b in input_values:
                    table.add_row([a, b, function(a, b)])
            print(f"Truth Table for {function.__name__}:\n")
            print(table)
            print("\n")

    def visualize(self, increment_scale=0.001, save_image=True, image_name="composite_image.png"):
        images = []

        for function in self.functions:
            if function.__code__.co_argcount == 1:
                input_range = np.arange(0, 1 + increment_scale, increment_scale)
                output_values = [function(x) for x in input_range]

                plt.figure(figsize=(8, 6))
                plt.plot(input_range, output_values, label='Function Output')
                plt.xlabel('Input')
                plt.ylabel('Output')
                plt.title(f'{function.__name__} - Line Graph')
                plt.legend()
                plt.grid(True)

                # Save the current plot to an in-memory buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Open the image and append it to the list of images
                images.append(Image.open(buffer))

            elif function.__code__.co_argcount == 2:
                input_range = np.arange(0, 1 + increment_scale, increment_scale)
                input_values = [(x, y) for x in input_range for y in input_range]
                output_values = np.array([function(x, y) for x, y in input_values])

                grid_size = int(1 / increment_scale) + 1
                input_values = np.array(input_values).reshape(grid_size, grid_size, -1)
                output_values = output_values.reshape(grid_size, grid_size)

                plt.figure(figsize=(10, 8))
                plt.imshow(output_values, extent=(0, 1, 0, 1), vmax = 1 , vmin = 0, origin='lower', cmap='RdYlGn')
                plt.colorbar()
                plt.xlabel('Input A')
                plt.ylabel('Input B')
                plt.title(f'{function.__name__} - Heatmap')

                # Save the current plot to an in-memory buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Open the image and append it to the list of images
                images.append(Image.open(buffer))

            elif function.__code__.co_argcount == 3:
                input_range = np.arange(0, 1 + increment_scale, increment_scale)
                input_values = [(x, y, z) for x in input_range for y in input_range for z in input_range]
                output_values = np.array([function(x, y, z) for x, y, z in input_values])

                grid_size = int(1 / increment_scale) + 1
                input_values = np.array(input_values).reshape(grid_size, grid_size, grid_size, -1)
                output_values = output_values.reshape(grid_size, grid_size, grid_size)

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                x, y, z = input_values[:, :, :, 0], input_values[:, :, :, 1], input_values[:, :, :, 2]
                ax.scatter(x, y, z, c=output_values, cmap='viridis')
                ax.set_xlabel('Input A')
                ax.set_ylabel('Input B')
                ax.set_zlabel('Input C')
                ax.set_title(f'{function.__name__} - 3D Plot')

                # Save the current plot to an in-memory buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Open the image and append it to the list of images
                images.append(Image.open(buffer))

            else:
                raise ValueError(f"Unsupported number of function parameters for function {function.__name__}")

            plt.close()  # Close the current plot to prevent displaying it

        # Calculate the number of rows and columns for the square arrangement
        num_images = len(images)
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))

        # Calculate the size of the composite image
        composite_width = cols * images[0].width
        composite_height = rows * images[0].height

        # Create a composite image
        composite_image = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))

        # Paste each image into the composite image
        for i, img in enumerate(images):
            x_offset = (i % cols) * images[0].width
            y_offset = (i // cols) * images[0].height
            composite_image.paste(img, (x_offset, y_offset))

        # Display the composite image
        plt.figure(figsize=(10, 8))
        plt.imshow(composite_image)
        plt.axis('off')  # Turn off axes
        plt.show()

        # Save the composite image to the working directory
        if save_image:
            save_path = os.path.join(os.getcwd(), image_name)
            composite_image.save(save_path)
            print(f"Composite image saved as '{image_name}' in the working directory.")

class Operands:
    class NullaryOps:
        def logical_true(a=None):
            """
            Logical True operator.

            Args:
                a (bool or float, optional): Input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                int or bool: Returns 1 for numerical input (int or float), True for Boolean input True, and 1 if no input is provided.
            """
            if isinstance(a, bool):
                return True
            elif isinstance(a, (int, float)):
                return 1
            else:
                return 1
        def logical_false(a=None):
            """
            Logical False operator.

            Args:
                a (bool or float, optional): Input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                int or bool: Returns 0 for numerical input (int or float), False for Boolean input False, and 0 if no input is provided.
            """
            if isinstance(a, bool):
                return False
            elif isinstance(a, (int, float)):
                return 0
            else:
                return 0
    class UnaryOps:
        def logical_identity(a):
            """
            Logical Identity operator.

            Args:
                a (bool or float): Input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the input value itself.
            """
            if isinstance(a, (bool, int, float)):
                return a
            else:
                return None  # or raise an error if needed
        def logical_negation(a):
            """
            Logical Negation (NOT) operator.

            Args:
                a (bool or float): Input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the negation of the input value. For a Boolean input, it returns the opposite truth value (True -> False, False -> True).
                For a decimal input, it returns 1.0 minus the input value.
            """
            if isinstance(a, bool):
                return not a
            elif isinstance(a, (int, float)):
                return 1.0 - a
            else:
                return None  # or raise an error if needed
    class BinaryOps:
        #TODO: go over each operation and decide if the equation is correct
        def logical_contradiction(a, b):
            """
            Contradiction (0) binary operator.

            Args:
                p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Always returns False (0) regardless of the input values.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return False
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return 0.0
            else:
                return None  # or raise an error if needed
        def logical_nor(a, b):
            """
            Logical NOR operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns True if both inputs are False, otherwise False.
                For decimal inputs, it returns the complement of the sum of the two input values (1 - a - b).
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return not (a or b)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return (1.0 - a) * (1.0 - b)
            else:
                return None  # or raise an error if needed
        def converse_nonimplication(a, b):
            """
            Converse Nonimplication (↚) operator.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns True if p is False and b is True. Otherwise, returns False.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return (not a) and b
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return (1.0 - a) * b
            else:
                return None  # or raise an error if needed
        def binary_negation_a(a, b):
            """
            Binary Negation operator.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, same type as 'a'.

            Returns:
                bool or float: Returns the negation of the first input 'a'.
            """
            if isinstance(a, bool):
                return not a
            elif isinstance(a, (int, float)):
                return 1.0 - a
            else:
                # Handle the case where 'a' is of an unsupported type
                raise ValueError("Input 'a' must be a bool, int, or float.")     
        def logical_xor(a, b):
            """
            Logical Exclusive Disjunction (XOR) operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns True if exactly one of the inputs is True, and False if both are True or both are False.
                For decimal inputs, it returns the result of (a + b - 2 * a * b).
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return (a and not b) or (not a and b)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a + b - 2 * a * b
            else:
                return None  # or raise an error if needed
        def material_nonimplication(a, b):
            """
            Material Nonimplication operator.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the result of the material nonimplication operation, which is true if 'a' is true and 'b' is false,
                and false otherwise.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return a and (not b)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a * (1.0 - b)
            else:
                raise ValueError("Inputs 'a' and 'b' must be bool, int, or float.")
        def binary_negation_b(a, b):
            """
            Binary Negation (NOT) operator.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the negation of the second input value. For Boolean inputs, it returns the opposite truth value (True -> False, False -> True).
                For decimal inputs, it returns 1.0 minus the second input value.
            """
            if isinstance(b, bool):
                return not b
            elif isinstance(b, (int, float)):
                return 1.0 - b
            else:
                return None  # or raise an error if neededdef logical_xor(a, b):
        def logical_nand(a, b):
            """
            Logical NAND operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns True if either of the inputs is False, otherwise False.
                For decimal inputs, it returns the complement of the product of the two input values (1 - a * b).
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return not (a and b)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return 1.0 - a * b
            else:
                return None  # or raise an error if needed
        def logical_conjunction(a, b):
            """
            Logical Conjunction (AND) operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the logical conjunction of the two input values. For Boolean inputs, it returns True if both inputs are True, otherwise False.
                For decimal inputs, it returns the product of the two input values.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return a and b
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a * b
            else:
                return None  # or raise an error if needed
        def logical_equality(a, b):
            """
            Logical Equality (XNOR) operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns True if both inputs are equal (either both True or both False), otherwise returns False.
                For decimal inputs, it returns 1.0 if both inputs are equal, and 0.0 if they are not equal.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return a == b
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                # return 1 - abs(a - b)
                return a * b + (1 - a) * (1 - b)
            else:
                return None  # or raise an error if needed
        def binary_projection_b(a, b):
            """
            Binary Operation: Projection function b.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the second input value (b) without any modification.
            """
            if isinstance(b, bool):
                return b
            elif isinstance(b, (int, float)):
                return b
            else:
                return None  # or raise an error if needed
        def material_implication(a, b):
            """
            Material Implication (→) operator.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the result of the material implication operation.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return not a or b
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return 1.0 - a + a * b
            else:
                return None  # or raise an error if needed
        def binary_projection_a(a, b):
            """
            Projection function P.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the value of the first input, a.
            """
            if isinstance(a, bool):
                return a
            elif isinstance(a, (int, float)):
                return a
            else:
                return None  # or raise an error if needed
        def converse_implication(a, b):
            """
            Converse Implication (Material Implication) operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns True if b is False or a is True. Returns False if b is True and a is False.
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return not b or a
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return 1.0 - (b * (1.0 - a))
            else:
                return None  # or raise an error if needed
        def logical_disjunction(a, b):
            """
            Logical Disjunction (OR) operator.

            Args:
                a (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Returns the logical disjunction of the two input values. For Boolean inputs, it returns True if at least one input is True, otherwise False.
                For decimal inputs, it returns the sum of the two input values minus their product (a + b - a * b).
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return a or b
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a + b - a * b
            else:
                return None  # or raise an error if needed
        def logical_tautology(a, b):
            """
            Tautology (⊤) operator.

            Args:
                a (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                b (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: If both inputs are Boolean, returns True (bool) regardless of the input values. If one or both inputs are decimal, returns 1.0 (float).
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return True
            else:
                return 1.0

                return None
    class TestBinaryOp:
        def templateBinaryFunction(a, b):
            """
            Does Nothing.

            Args:
                p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Always returns Null
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return None
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return None
            else:
                None
        def a_plus_b(a, b):
            """
            A plus B.

            Args:
                p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Always returns Null
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return a + b 
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a + b 
            else:
                None        
        def a_times_b(a, b):
            """
            Does Nothing.

            Args:
                p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Always returns Null
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return None
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return None
            else:
                None
        def expression1(a, b):
            """
             (a * b) / (a + b).

            Args:
                p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
                q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

            Returns:
                bool or float: Always returns Null
            """
            if isinstance(a, bool) and isinstance(b, bool):
                return  (a - 1) / (b + 1) + 1
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return  (a * b) + ((1 - a) * (1 - b))
            else:
                None
        

#Test Operations
# operands_list = [
#     Operands.BinaryOps.logical_contradiction,
#     Operands.BinaryOps.logical_nor,
#     Operands.BinaryOps.converse_nonimplication,
#     Operands.BinaryOps.binary_negation_a,
#     Operands.BinaryOps.logical_xor,
#     Operands.BinaryOps.material_nonimplication,
#     Operands.BinaryOps.binary_negation_b,
#     Operands.BinaryOps.logical_nand,
#     Operands.BinaryOps.logical_conjunction,
#     Operands.BinaryOps.logical_equality,
#     Operands.BinaryOps.binary_projection_b,
#     Operands.BinaryOps.material_implication,
#     Operands.BinaryOps.binary_projection_a,
#     Operands.BinaryOps.converse_implication,
#     Operands.BinaryOps.logical_disjunction,
#     Operands.BinaryOps.logical_tautology
# ]

# visualizer = FunctionVisualizer(operands_list)
# visualizer.truth_table("boolean")
# visualizer.visualize()
# 8,4,2,1
operands_list = [
    
    Operands.BinaryOps.logical_conjunction,
    Operands.BinaryOps.material_nonimplication,
    Operands.BinaryOps.converse_nonimplication,
    Operands.BinaryOps.logical_nor
    
]
visualizer = FunctionVisualizer(operands_list)
visualizer.truth_table(input_choices="boolean")
visualizer.visualize(image_name="FromZero.png")

# 7,11,13,14
operands_list = [
    
    Operands.BinaryOps.logical_nand,
    Operands.BinaryOps.material_implication,
    Operands.BinaryOps.converse_implication,
    Operands.BinaryOps.logical_disjunction
    
]
visualizer = FunctionVisualizer(operands_list)
visualizer.truth_table(input_choices="boolean")
visualizer.visualize(image_name="FromOne.png")
# 6,9
operands_list = [
    
    Operands.BinaryOps.logical_xor,
    Operands.BinaryOps.logical_equality,
    
]
visualizer = FunctionVisualizer(operands_list)
visualizer.truth_table(input_choices="boolean")
visualizer.visualize(image_name="insideOutside.png")

# the inbetweens  - 10,3,5,12
operands_list = [
    
    Operands.BinaryOps.binary_projection_b,
    Operands.BinaryOps.binary_negation_a,
    Operands.BinaryOps.binary_negation_b,
    Operands.BinaryOps.binary_projection_a
]
visualizer = FunctionVisualizer(operands_list)
visualizer.truth_table(input_choices="boolean")
visualizer.visualize(image_name="inBetweens.png")

operands_list = [
    
    Operands.BinaryOps.logical_tautology,
    Operands.BinaryOps.logical_contradiction
]
visualizer = FunctionVisualizer(operands_list)
visualizer.truth_table(input_choices="boolean")
visualizer.visualize(image_name="TrueFalse.png")