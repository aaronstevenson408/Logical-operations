#TODO visualization of the different operations , (truth tables, line graphs (for unary), and heat maps(for binary))


# Nullary operations

def logical_true(a):
    """
    Logical True operator.
    
    Args:
        a (bool or float): Input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns True for Boolean input True or decimal input 1.0, and False for all other inputs.
    """
    if isinstance(a, bool):
        return a is True
    elif isinstance(a, (int, float)):
        return a == 1.0
    else:
        return False
def test_logical_true():
    """
    Test function for the logical True operator.
    """
    test_cases = [True, False, 1.0, 0.0, 0.5, "True"]
    
    for case in test_cases:
        result = logical_true(case)
        print(f"Input: {case}, Output: {result}")
        
def logical_false(a):
    """
    Logical False operator.

    Args:
        a (bool or float): Input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns False for Boolean input False or decimal input 0.0, and True for all other inputs.
    """
    if isinstance(a, bool):
        return a is False
    elif isinstance(a, (int, float)):
        return a == 0.0
    else:
        return True
def test_logical_false():
    """
    Test function for the logical False operator.
    """
    test_cases = [True, False, 1.0, 0.0, 0.5, "False"]

    for case in test_cases:
        result = logical_false(case)
        print(f"Input: {case}, Output: {result}")


# Unary operations

# 0 (T,F)(p) p Logical identity
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
def test_logical_identity():
    """
    Test function for the Logical Identity operator.
    """
    test_cases = [True, False, 1.0, 0.0, 0.5, "True", "False"]

    for case in test_cases:
        result = logical_identity(case)
        print(f"Input: {case}, Output: {result}")

# 1 (F,T)(p) ¬p, Np, Fpq, or ~p Logical  negation
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
def test_logical_negation():
    """
    Test function for the Logical Negation (NOT) operator.
    """
    test_cases = [True, False, 1.0, 0.0, 0.5, "True", "False"]

    for case in test_cases:
        result = logical_negation(case)
        print(f"Input: {case}, Output: {result}")
    
    for value in range(0, 105, 5):
        a = value / 100.0  # Convert the integer value to a decimal value between 0 and 1
        result = logical_negation(a)
        print(f"logical_negation({a}) = {result}")
        
#Binary operations

# 0	(F F F F)(p, q)	⊥	false, Opq	Contradiction
def logical_contradiction(p, q):
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
def test_logical_contradiction():
    """
    Test function for the Contradiction (0) binary operator.
    """
    test_cases = [(True, True), (True, False), (False, True), (False, False), (1.0, 0.0), (0.5, 0.5)]

    for p, q in test_cases:
        result = logical_contradiction(p, q)
        print(f"Input: ({p}, {q}), Output: {result}")

# 1	(F F F T)(p, q)	NOR	p ↓ q, Xpq	Logical NOR
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
        return 1.0 - a - b
    else:
        return None  # or raise an error if needed
def test_logical_nor():
    """
    Test function for the Logical NOR operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    for a, b in test_cases:
        result = logical_nor(a, b)
        print(f"Input: ({a}, {b}), Output: {result}")

# 2	(F F T F)(p, q)	↚	p ↚ q, Mpq	Converse nonimplication
def converse_nonimplication(p, q):
    """
    Converse Nonimplication (↚) operator.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns True if p is False and q is True. Otherwise, returns False.
    """
    if isinstance(p, bool) and isinstance(q, bool):
        return (not p) and q
    elif isinstance(p, (int, float)) and isinstance(q, (int, float)):
        return (1.0 - p) * q
    else:
        return None  # or raise an error if needed
def test_converse_nonimplication():
    """
    Test function for the Converse Nonimplication (↚) operator.
    """
    test_cases = [
        (True, True), (True, False), (False, True),
        (1.0, 1.0), (0.0, 1.0), (0.5, 0.5), ("True", "False")
    ]

    for p, q in test_cases:
        result = converse_nonimplication(p, q)
        print(f"Input: {p}, {q}, Output: {result}")

# 3	(F F T T)(p, q)	¬p, ~p	¬p, Np, Fpq	Negation P
def binary_negation_p(p, q):
    """
    Binary Negation operator.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, same type as 'p'.

    Returns:
        bool or float: Returns the negation of the first input 'p'.
    """
    if isinstance(p, bool):
        return not p
    elif isinstance(p, (int, float)):
        return 1.0 - p
    else:
        return None  # or raise an error if needed
def test_binary_negation_p():
    """
    Test function for the Binary Negation operator.
    """
    test_cases = [(True, True), (False, True), (True, False), (False, False), (1.0, 0.5), (0.0, 0.0), ("True", "False")]

    for case in test_cases:
        result = binary_negation_p(case[0], case[1])
        print(f"Input: {case[0]}, {case[1]}, Output: {result}")

# 4 (F T F F)(p, q)	↛	p ↛ q, Lpq	Material nonimplication
def material_nonimplication(A, B):
    """
    Material Nonimplication operator.

    Args:
        A (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        B (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns the result of the material nonimplication operation, which is true if A is true and B is false,
        and false otherwise.
    """
    if isinstance(A, bool) and isinstance(B, bool):
        return A and (not B)
    elif isinstance(A, (int, float)) and isinstance(B, (int, float)):
        return A * (1.0 - B)
    else:
        return None  # or raise an error if needed
def test_material_nonimplication():
    """
    Test function for the Material Nonimplication operator (A && (!B)).
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
        ("True", "False"),
    ]

    for case in test_cases:
        result = material_nonimplication(case[0], case[1])
        print(f"Input: ({case[0]}, {case[1]}), Output: {result}")

# 5	(F T F T)(p, q)	¬q, ~q	¬q, Nq, Gpq	Negation Q
def binary_negation_q(p, q):
    """
    Binary Negation (NOT) operator.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns the negation of the second input value. For Boolean inputs, it returns the opposite truth value (True -> False, False -> True).
        For decimal inputs, it returns 1.0 minus the second input value.
    """
    if isinstance(q, bool):
        return not q
    elif isinstance(q, (int, float)):
        return 1.0 - q
    else:
        return None  # or raise an error if needed
def test_binary_negation_q():
    """
    Test function for the Binary Negation (NOT) operator.
    """
    test_cases = [(True, True), (False, True), (1.0, 0.0), (0.5, 0.5), (True, 0.5)]

    for case in test_cases:
        result = binary_negation_q(case[0], case[1])
        print(f"Input: ({case[0]}, {case[1]}), Output: {result}")

# 6	(F T T F)(p, q)	XOR	p ⊕ q, Jpq	Exclusive disjunction
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
def test_logical_xor():
    """
    Test function for the Logical Exclusive Disjunction (XOR) operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    for a, b in test_cases:
        result = logical_xor(a, b)
        print(f"Input: ({a}, {b}), Output: {result}")

# 7	(F T T T)(p, q)	NAND	p ↑ q, Dpq	Logical NAND
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
def test_logical_nand():
    """
    Test function for the Logical NAND operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    for a, b in test_cases:
        result = logical_nand(a, b)
        print(f"Input: ({a}, {b}), Output: {result}")

# 8	(T F F F)(p, q)	AND	p ∧ q, Kpq	Logical conjunction
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
def test_logical_conjunction():
    """
    Test function for the Logical Conjunction (AND) operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    for a, b in test_cases:
        result = logical_conjunction(a, b)
        print(f"Input: ({a}, {b}), Output: {result}")

# 9	(T F F T)(p, q)	XNOR	p If and only if q, Epq	Logical biconditional
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
        return 1 - abs(a - b)
    else:
        return None  # or raise an error if needed
def test_logical_equality():
    """
    Test function for the Logical Equality (XNOR) operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    for a, b in test_cases:
        result = logical_equality(a, b)
        print(f"Input: ({a}, {b}), Output: {result}")

# 10 (T F T F)(p, q)	q	q, Hpq	Projection function Q
def binary_projection_q(p, q):
    """
    Binary Operation: Projection function Q.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns the second input value (q) without any modification.
    """
    if isinstance(p, bool):
        return q
    elif isinstance(p, (int, float)):
        return q
    else:
        return None  # or raise an error if needed
def test_binary_projection_q():
    """
    Test function for the Binary Operation: Projection function Q.
    """
    test_cases = [(True, True), (True, False), (False, True), (False, False), (1.0, 0.5), (0.0, 1.0)]

    for case in test_cases:
        result = binary_projection_q(*case)
        print(f"Input: ({case[0]}, {case[1]}), Output: {result}")

# 11 (T F T T)(p, q)	p → q	if p then q, Cpq	Material implication
def material_implication(p, q):
    """
    Material Implication (→) operator.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns the result of the material implication operation.
    """
    if isinstance(p, bool) and isinstance(q, bool):
        return not p or q
    elif isinstance(p, (int, float)) and isinstance(q, (int, float)):
        return 1.0 - p + p * q
    else:
        return None  # or raise an error if needed
def test_material_implication():
    """
    Test function for the Material Implication (→) operator.
    """
    test_cases = [(True, True), (True, False), (False, True), (False, False), (0.5, 0.8), (0.0, 0.0), (1.0, 1.0),(0.0, 1.0), (1.0, 0.0)]

    for p, q in test_cases:
        result = material_implication(p, q)
        print(f"Input: (p={p}, q={q}), Output: {result}")

# 12 (T T F F)(p, q)	p	p, Ipq	Projection Function P
def projection_function_p(p, q):
    """
    Projection function P.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns the value of the first input, p.
    """
    if isinstance(p, bool):
        return p
    elif isinstance(p, (int, float)):
        return p
    else:
        return None  # or raise an error if needed
def test_projection_function_p():
    """
    Test function for the Projection function P.
    """
    test_cases = [(True, True), (True, False), (False, True), (False, False)]

    for p, q in test_cases:
        result = projection_function_p(p, q)
        print(f"Input: (p={p}, q={q}), Output: {result}")

# 13 (T T F T)(p, q) p ← q	p if q, Bpq	Converse implication 
def converse_implication(p, q):
    """
    Converse Implication (Material Implication) operator.

    Args:
        p (bool or float): The first input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): The second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: Returns True if q is False or p is True. Returns False if q is True and p is False.
    """
    if isinstance(p, bool) and isinstance(q, bool):
        return not q or p
    elif isinstance(p, (int, float)) and isinstance(q, (int, float)):
        return 1.0 - (q * (1.0 - p))
    else:
        return None  # or raise an error if needed
def test_converse_implication():
    """
    Test function for the Converse Implication (Material Implication) operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]

    for case in test_cases:
        p, q = case
        result = converse_implication(p, q)
        print(f"Input: ({p}, {q}), Output: {result}")

# 14 (T T T F)(p, q) OR	p ∨ q, Apq	Logical disjunction
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
def test_logical_disjunction():
    """
    Test function for the Logical Disjunction (OR) operator.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
    ]

    for a, b in test_cases:
        result = logical_disjunction(a, b)
        print(f"Input: ({a}, {b}), Output: {result}")

# 15	(T T T T)(p, q)	⊤	true, Vpq	Tautology
def logical_tautology(p, q):
    """
    Tautology (⊤) operator.

    Args:
        p (bool or float): First input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).
        q (bool or float): Second input value, can be either a Boolean (True/False) or a decimal (0.0-1.0).

    Returns:
        bool or float: If both inputs are Boolean, returns True (bool) regardless of the input values. If one or both inputs are decimal, returns 1.0 (float).
    """
    if isinstance(p, bool) and isinstance(q, bool):
        return True
    else:
        return 1.0
def test_logical_tautology():
    """
    Test function for the Tautology (⊤) operator.
    """
    test_cases = [(True, True), (True, False), (False, True), (False, False), (1.0, 0.5), (0.0, 0.0)]

    for p, q in test_cases:
        result = logical_tautology(p, q)
        print(f"Input: ({p}, {q}), Output: {result}")
