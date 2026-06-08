"""
builtintools/calculator.py — Math calculator tool for LangGraph / LangChain agents.

Provides @tool-decorated functions covering basic through moderate maths:

  Basic arithmetic
  ─────────────────
  - calculate          : Evaluate a safe arithmetic expression string
  - basic_arithmetic   : Add / subtract / multiply / divide two numbers

  Number utilities
  ────────────────
  - power_and_roots    : Exponentiation, square-root, nth-root, logarithm
  - percentage         : Percentage, percentage change, percentage of total

  Algebra helpers
  ───────────────
  - gcd_lcm            : Greatest common divisor and least common multiple
  - prime_check        : Primality test and prime factorisation
  - factorial_and_combo: Factorial, permutations (nPr) and combinations (nCr)

  Statistics
  ──────────
  - statistics_basic   : Mean, median, mode, variance, standard deviation

  Geometry
  ────────
  - geometry           : Area and perimeter / volume for common shapes

  Unit conversion
  ───────────────
  - unit_convert       : Length, weight, temperature, and speed conversions

Usage in your LangGraph agent
------------------------------
    from builtintools.calculator import TOOLS
    model = ChatGroq(...).bind_tools(TOOLS)

All tools are pure Python (stdlib only) — no extra dependencies required.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Annotated, Literal
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Safe operators allowed in expression evaluation
_SAFE_OPS: dict = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}

_SAFE_NAMES: dict = {
    "pi":    math.pi,
    "e":     math.e,
    "tau":   math.tau,
    "inf":   math.inf,
    "sqrt":  math.sqrt,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "asin":  math.asin,
    "acos":  math.acos,
    "atan":  math.atan,
    "atan2": math.atan2,
    "ceil":  math.ceil,
    "floor": math.floor,
    "round": round,
    "abs":   abs,
    "exp":   math.exp,
    "factorial": math.factorial,
    "gcd":   math.gcd,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only whitelisted ops and names."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            val = _SAFE_NAMES[node.id]
            if callable(val):
                raise ValueError(
                    f"'{node.id}' is a function — call it with arguments, e.g. {node.id}(x)."
                )
            return float(val)
        raise ValueError(f"Unknown name '{node.id}'. "
                         f"Allowed names: {', '.join(sorted(_SAFE_NAMES))}.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed (e.g. sqrt(x)).")
        func_name = node.func.id
        if func_name not in _SAFE_NAMES or not callable(_SAFE_NAMES[func_name]):
            raise ValueError(f"Function '{func_name}' is not allowed.")
        args = [_safe_eval(a) for a in node.args]
        return _SAFE_NAMES[func_name](*args)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Operator '{op_type.__name__}' is not supported.")
        left  = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type is ast.Div and right == 0:
            raise ZeroDivisionError("Division by zero.")
        if op_type is ast.FloorDiv and right == 0:
            raise ZeroDivisionError("Floor division by zero.")
        if op_type is ast.Mod and right == 0:
            raise ZeroDivisionError("Modulo by zero.")
        return _SAFE_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unary operator '{op_type.__name__}' is not supported.")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))

    raise ValueError(f"Unsupported expression element: {type(node).__name__}.")


def _pretty(value: float) -> float | int:
    """Return int if value is a whole number, else float rounded to 10 dp."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return round(float(value), 10)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def calculate(expression: str) -> dict:
    """Safely evaluate an arithmetic / mathematical expression string.

    This is the go-to tool for most calculations. It supports standard
    operators (+, -, *, /, //, %, **) and common math functions and constants.

    Supported constants:  pi, e, tau, inf
    Supported functions:  sqrt, log, log2, log10, sin, cos, tan,
                          asin, acos, atan, atan2, ceil, floor,
                          round, abs, exp, factorial, gcd

    Args:
        expression: A mathematical expression as a string.
                    Examples:
                      "2 + 3 * 4"          → 14
                      "sqrt(144)"          → 12
                      "pi * 5**2"          → 78.5398…
                      "log(1000, 10)"      → 3.0
                      "factorial(6)"       → 720
                      "gcd(48, 18)"        → 6
                      "(3 + 4j)"           → not supported (complex not allowed)

    Returns:
        dict with keys:
          - expression (str): The original expression
          - result (int | float): The computed numeric result
          - error (str | None): Error message if evaluation failed, else null
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return {"expression": expression, "result": _pretty(result), "error": None}
    except ZeroDivisionError as exc:
        return {"expression": expression, "result": None, "error": f"ZeroDivisionError: {exc}"}
    except (ValueError, TypeError, SyntaxError, OverflowError) as exc:
        return {"expression": expression, "result": None, "error": str(exc)}


@tool
def basic_arithmetic(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide", "floor_divide", "modulo", "power"],
) -> dict:
    """Perform a single arithmetic operation on two numbers.

    Use this when you have two explicit numbers and a clear operation in mind,
    rather than a full expression string.

    Args:
        a: The first operand (left-hand side).
        b: The second operand (right-hand side).
        operation: One of:
            - "add"          : a + b
            - "subtract"     : a - b
            - "multiply"     : a * b
            - "divide"       : a / b  (true division, returns float)
            - "floor_divide" : a // b (integer division)
            - "modulo"       : a % b  (remainder)
            - "power"        : a ** b (exponentiation)

    Returns:
        dict with keys:
          - a, b (float): The input operands
          - operation (str): The chosen operation
          - expression (str): Human-readable expression (e.g. "6 + 4")
          - result (int | float): The computed result
          - error (str | None): Error message on failure, else null
    """
    ops = {
        "add":          (operator.add,       "+"),
        "subtract":     (operator.sub,       "-"),
        "multiply":     (operator.mul,       "×"),
        "divide":       (operator.truediv,   "÷"),
        "floor_divide": (operator.floordiv,  "//"),
        "modulo":       (operator.mod,       "%"),
        "power":        (operator.pow,       "**"),
    }
    if operation not in ops:
        return {"a": a, "b": b, "operation": operation, "expression": "",
                "result": None, "error": f"Unknown operation '{operation}'."}

    fn, symbol = ops[operation]
    expr = f"{a} {symbol} {b}"
    try:
        if operation in ("divide", "floor_divide", "modulo") and b == 0:
            raise ZeroDivisionError(f"Cannot {operation} by zero.")
        result = fn(a, b)
        return {"a": a, "b": b, "operation": operation, "expression": expr,
                "result": _pretty(result), "error": None}
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        return {"a": a, "b": b, "operation": operation, "expression": expr,
                "result": None, "error": str(exc)}


@tool
def power_and_roots(
    value: float,
    operation: Literal["square", "cube", "sqrt", "cbrt", "nth_root", "log", "log2", "log10", "exp"],
    n: float = 2.0,
    base: float = math.e,
) -> dict:
    """Compute powers, roots, exponentials, and logarithms of a number.

    Args:
        value: The input number to operate on.
        operation: One of:
            - "square"   : value²
            - "cube"     : value³
            - "sqrt"     : √value (square root)
            - "cbrt"     : ∛value (cube root)
            - "nth_root" : value^(1/n) — uses the 'n' parameter
            - "log"      : log(value, base) — uses the 'base' parameter (default: e → natural log)
            - "log2"     : log₂(value)
            - "log10"    : log₁₀(value)
            - "exp"      : e^value
        n: Exponent for nth_root (default 2). Example: n=4 → 4th root.
        base: Logarithm base for "log" operation (default: math.e for natural log).

    Returns:
        dict with keys:
          - value (float): Input number
          - operation (str): The operation performed
          - result (int | float): The computed result
          - error (str | None): Error message on failure, else null
    """
    try:
        match operation:
            case "square":   result = value ** 2
            case "cube":     result = value ** 3
            case "sqrt":
                if value < 0:
                    raise ValueError("Cannot take square root of a negative number.")
                result = math.sqrt(value)
            case "cbrt":     result = math.copysign(abs(value) ** (1/3), value)
            case "nth_root":
                if n == 0:
                    raise ValueError("Root degree n cannot be zero.")
                result = math.copysign(abs(value) ** (1/n), value)
            case "log":
                if value <= 0:
                    raise ValueError("Logarithm requires a positive input.")
                if base <= 0 or base == 1:
                    raise ValueError("Logarithm base must be > 0 and ≠ 1.")
                result = math.log(value, base)
            case "log2":
                if value <= 0:
                    raise ValueError("log2 requires a positive input.")
                result = math.log2(value)
            case "log10":
                if value <= 0:
                    raise ValueError("log10 requires a positive input.")
                result = math.log10(value)
            case "exp":      result = math.exp(value)
            case _:
                return {"value": value, "operation": operation, "result": None,
                        "error": f"Unknown operation '{operation}'."}
        return {"value": value, "operation": operation, "result": _pretty(result), "error": None}
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        return {"value": value, "operation": operation, "result": None, "error": str(exc)}


@tool
def percentage(
    operation: Literal["percent_of", "percent_change", "what_percent", "add_percent", "subtract_percent"],
    value: float,
    reference: float = 100.0,
) -> dict:
    """Perform common percentage calculations.

    Args:
        operation: One of:
            - "percent_of"       : What is `value`% of `reference`?
                                   e.g. value=20, reference=500 → 100
            - "percent_change"   : % change from `reference` to `value`.
                                   e.g. value=150, reference=100 → +50%
            - "what_percent"     : `value` is what % of `reference`?
                                   e.g. value=25, reference=200 → 12.5%
            - "add_percent"      : `reference` + `value`% of `reference`.
                                   e.g. value=10, reference=200 → 220
            - "subtract_percent" : `reference` - `value`% of `reference`.
                                   e.g. value=10, reference=200 → 180
        value: Primary number (percent amount or new value depending on operation).
        reference: Base / original value (default 100).

    Returns:
        dict with keys:
          - operation (str): Operation performed
          - value, reference (float): Input numbers
          - result (int | float): Computed result
          - unit (str): "%" when result is a percentage, "" otherwise
          - error (str | None): Error message on failure, else null
    """
    try:
        match operation:
            case "percent_of":
                result = (value / 100) * reference
                unit = ""
            case "percent_change":
                if reference == 0:
                    raise ZeroDivisionError("Reference (original value) cannot be zero.")
                result = ((value - reference) / abs(reference)) * 100
                unit = "%"
            case "what_percent":
                if reference == 0:
                    raise ZeroDivisionError("Reference cannot be zero.")
                result = (value / reference) * 100
                unit = "%"
            case "add_percent":
                result = reference + (value / 100) * reference
                unit = ""
            case "subtract_percent":
                result = reference - (value / 100) * reference
                unit = ""
            case _:
                return {"operation": operation, "value": value, "reference": reference,
                        "result": None, "unit": "", "error": f"Unknown operation '{operation}'."}
        return {"operation": operation, "value": value, "reference": reference,
                "result": _pretty(result), "unit": unit, "error": None}
    except (ZeroDivisionError, ValueError) as exc:
        return {"operation": operation, "value": value, "reference": reference,
                "result": None, "unit": "", "error": str(exc)}


@tool
def gcd_lcm(a: int, b: int) -> dict:
    """Compute the Greatest Common Divisor (GCD) and Least Common Multiple (LCM) of two integers.

    Useful for fraction simplification, scheduling problems, and number theory tasks.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        dict with keys:
          - a, b (int): Input integers
          - gcd (int): Greatest common divisor
          - lcm (int | None): Least common multiple (null if both inputs are 0)
          - error (str | None): Error message on failure, else null
    """
    try:
        a, b = int(a), int(b)
        g = math.gcd(abs(a), abs(b))
        lcm = (abs(a) * abs(b)) // g if g != 0 else None
        return {"a": a, "b": b, "gcd": g, "lcm": lcm, "error": None}
    except (ValueError, TypeError) as exc:
        return {"a": a, "b": b, "gcd": None, "lcm": None, "error": str(exc)}


@tool
def prime_check(n: int) -> dict:
    """Check if an integer is prime and return its full prime factorisation.

    Args:
        n: A positive integer to analyse.
           Note: Numbers ≤ 1 are not prime by definition.

    Returns:
        dict with keys:
          - n (int): The input number
          - is_prime (bool): True if n is a prime number
          - prime_factors (list[int]): Sorted list of prime factors (with repetition)
                                       e.g. 12 → [2, 2, 3]
          - unique_factors (list[int]): Unique prime factors e.g. 12 → [2, 3]
          - error (str | None): Error message on failure, else null
    """
    try:
        n = int(n)
        if n < 1:
            return {"n": n, "is_prime": False, "prime_factors": [],
                    "unique_factors": [], "error": "Input must be a positive integer ≥ 1."}

        def _is_prime(num: int) -> bool:
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for i in range(3, int(num**0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True

        def _factorise(num: int) -> list[int]:
            factors = []
            d = 2
            while d * d <= num:
                while num % d == 0:
                    factors.append(d)
                    num //= d
                d += 1
            if num > 1:
                factors.append(num)
            return factors

        factors = _factorise(n)
        return {
            "n":              n,
            "is_prime":       _is_prime(n),
            "prime_factors":  factors,
            "unique_factors": sorted(set(factors)),
            "error":          None,
        }
    except (ValueError, TypeError) as exc:
        return {"n": n, "is_prime": None, "prime_factors": [],
                "unique_factors": [], "error": str(exc)}


@tool
def factorial_and_combo(
    n: int,
    r: int = 0,
    operation: Literal["factorial", "permutation", "combination"] = "factorial",
) -> dict:
    """Compute factorial, permutations (nPr), or combinations (nCr).

    Args:
        n: Total number of items. Must be a non-negative integer.
        r: Number of items selected. Required for permutation and combination.
           Must satisfy 0 ≤ r ≤ n.
        operation: One of:
            - "factorial"   : n!  (uses only n, ignores r)
            - "permutation" : nPr = n! / (n-r)!
            - "combination" : nCr = n! / (r! × (n-r)!)

    Returns:
        dict with keys:
          - n, r (int): Input values
          - operation (str): Operation performed
          - result (int): Computed result
          - error (str | None): Error message on failure, else null
    """
    try:
        n, r = int(n), int(r)
        if n < 0:
            raise ValueError("n must be a non-negative integer.")
        if operation in ("permutation", "combination") and not (0 <= r <= n):
            raise ValueError(f"r must satisfy 0 ≤ r ≤ n. Got n={n}, r={r}.")

        match operation:
            case "factorial":
                result = math.factorial(n)
            case "permutation":
                result = math.factorial(n) // math.factorial(n - r)
            case "combination":
                result = math.comb(n, r)
            case _:
                return {"n": n, "r": r, "operation": operation, "result": None,
                        "error": f"Unknown operation '{operation}'."}

        return {"n": n, "r": r, "operation": operation, "result": result, "error": None}
    except (ValueError, OverflowError, TypeError) as exc:
        return {"n": n, "r": r, "operation": operation, "result": None, "error": str(exc)}


@tool
def statistics_basic(numbers: list[float]) -> dict:
    """Compute descriptive statistics for a list of numbers.

    Calculates: count, sum, min, max, range, mean (average), median,
    mode (most frequent value), variance, and standard deviation.

    Args:
        numbers: A non-empty list of numeric values.
                 Example: [4, 8, 15, 16, 23, 42]

    Returns:
        dict with keys:
          - count (int): Number of values
          - sum (float): Total sum
          - min (float): Minimum value
          - max (float): Maximum value
          - range (float): max − min
          - mean (float): Arithmetic average
          - median (float): Middle value when sorted
          - mode (float | list | None): Most common value(s); null if all values unique
          - variance (float): Population variance
          - std_dev (float): Population standard deviation
          - error (str | None): Error message on failure, else null
    """
    try:
        if not numbers:
            raise ValueError("The numbers list cannot be empty.")
        data = [float(x) for x in numbers]
        n = len(data)

        try:
            mode_val = statistics.mode(data)
            # multimode available in Python 3.8+
            multimode = statistics.multimode(data)
            mode_result = multimode if len(multimode) > 1 else mode_val
        except statistics.StatisticsError:
            mode_result = None

        return {
            "count":    n,
            "sum":      _pretty(sum(data)),
            "min":      _pretty(min(data)),
            "max":      _pretty(max(data)),
            "range":    _pretty(max(data) - min(data)),
            "mean":     _pretty(statistics.mean(data)),
            "median":   _pretty(statistics.median(data)),
            "mode":     mode_result,
            "variance": _pretty(statistics.pvariance(data)),
            "std_dev":  _pretty(statistics.pstdev(data)),
            "error":    None,
        }
    except (ValueError, TypeError) as exc:
        return {
            "count": 0, "sum": None, "min": None, "max": None, "range": None,
            "mean": None, "median": None, "mode": None,
            "variance": None, "std_dev": None, "error": str(exc),
        }


@tool
def geometry(
    shape: Literal[
        "circle", "rectangle", "square", "triangle", "trapezoid",
        "sphere", "cylinder", "cone", "cube", "cuboid"
    ],
    operation: Literal["area", "perimeter", "volume", "surface_area"],
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.0,
    h: float = 0.0,
    r: float = 0.0,
) -> dict:
    """Calculate geometric measurements (area, perimeter, volume, surface area).

    Parameter guide by shape
    ─────────────────────────────────────────────────────────────────────────
    circle      : r = radius
    rectangle   : a = width, b = height
    square      : a = side length
    triangle    : a, b, c = three side lengths  (Heron's formula for area)
    trapezoid   : a = parallel side 1, b = parallel side 2, h = height
    sphere      : r = radius
    cylinder    : r = radius, h = height
    cone        : r = base radius, h = height
    cube        : a = side length
    cuboid      : a = length, b = width, h = height
    ─────────────────────────────────────────────────────────────────────────

    Args:
        shape: The geometric shape (see list above).
        operation: One of "area", "perimeter", "volume", "surface_area".
        a, b, c: Side lengths / dimensions (usage depends on shape — see guide).
        h: Height (used for rectangle height, trapezoid, cylinder, cone, cuboid).
        r: Radius (used for circle, sphere, cylinder, cone).

    Returns:
        dict with keys:
          - shape (str): Shape name
          - operation (str): Measurement type
          - inputs (dict): The dimension values used
          - result (float): Computed measurement
          - unit_hint (str): Reminder that units match input (e.g. "square of input unit")
          - error (str | None): Error message on failure, else null
    """
    inputs = {"a": a, "b": b, "c": c, "h": h, "r": r}
    try:
        result: float | None = None
        unit_hint = ""

        match shape:
            case "circle":
                match operation:
                    case "area":
                        result = math.pi * r ** 2
                        unit_hint = "square of input unit"
                    case "perimeter":
                        result = 2 * math.pi * r
                        unit_hint = "input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for circle.")

            case "rectangle":
                match operation:
                    case "area":
                        result = a * b
                        unit_hint = "square of input unit"
                    case "perimeter":
                        result = 2 * (a + b)
                        unit_hint = "input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for rectangle.")

            case "square":
                match operation:
                    case "area":
                        result = a ** 2
                        unit_hint = "square of input unit"
                    case "perimeter":
                        result = 4 * a
                        unit_hint = "input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for square.")

            case "triangle":
                s = (a + b + c) / 2
                if a <= 0 or b <= 0 or c <= 0:
                    raise ValueError("All triangle sides must be positive.")
                if s - a <= 0 or s - b <= 0 or s - c <= 0:
                    raise ValueError("The given sides do not form a valid triangle.")
                match operation:
                    case "area":
                        result = math.sqrt(s * (s - a) * (s - b) * (s - c))
                        unit_hint = "square of input unit"
                    case "perimeter":
                        result = a + b + c
                        unit_hint = "input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for triangle.")

            case "trapezoid":
                match operation:
                    case "area":
                        result = 0.5 * (a + b) * h
                        unit_hint = "square of input unit"
                    case "perimeter":
                        raise ValueError(
                            "Trapezoid perimeter needs all 4 sides. "
                            "Use calculate() with the explicit formula instead."
                        )
                    case _:
                        raise ValueError(f"'{operation}' is not defined for trapezoid.")

            case "sphere":
                match operation:
                    case "area" | "surface_area":
                        result = 4 * math.pi * r ** 2
                        unit_hint = "square of input unit"
                    case "volume":
                        result = (4/3) * math.pi * r ** 3
                        unit_hint = "cubic of input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for sphere.")

            case "cylinder":
                match operation:
                    case "area" | "surface_area":
                        result = 2 * math.pi * r * (r + h)
                        unit_hint = "square of input unit"
                    case "volume":
                        result = math.pi * r ** 2 * h
                        unit_hint = "cubic of input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for cylinder.")

            case "cone":
                slant = math.sqrt(r ** 2 + h ** 2)
                match operation:
                    case "area" | "surface_area":
                        result = math.pi * r * (r + slant)
                        unit_hint = "square of input unit"
                    case "volume":
                        result = (1/3) * math.pi * r ** 2 * h
                        unit_hint = "cubic of input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for cone.")

            case "cube":
                match operation:
                    case "area" | "surface_area":
                        result = 6 * a ** 2
                        unit_hint = "square of input unit"
                    case "volume":
                        result = a ** 3
                        unit_hint = "cubic of input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for cube.")

            case "cuboid":
                match operation:
                    case "area" | "surface_area":
                        result = 2 * (a * b + b * h + a * h)
                        unit_hint = "square of input unit"
                    case "volume":
                        result = a * b * h
                        unit_hint = "cubic of input unit"
                    case _:
                        raise ValueError(f"'{operation}' is not defined for cuboid.")

            case _:
                raise ValueError(f"Unknown shape '{shape}'.")

        return {
            "shape": shape, "operation": operation,
            "inputs": inputs, "result": _pretty(result),
            "unit_hint": unit_hint, "error": None,
        }

    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return {
            "shape": shape, "operation": operation,
            "inputs": inputs, "result": None,
            "unit_hint": "", "error": str(exc),
        }


@tool
def unit_convert(
    value: float,
    from_unit: str,
    to_unit: str,
    category: Literal["length", "weight", "temperature", "speed"] = "length",
) -> dict:
    """Convert a value between common units of measurement.

    Supported units per category
    ─────────────────────────────────────────────────────────────────────────
    length      : mm, cm, m, km, inch, ft, yd, mile, nautical_mile
    weight      : mg, g, kg, tonne, oz, lb, stone
    temperature : C, F, K      (Celsius, Fahrenheit, Kelvin)
    speed       : m/s, km/h, mph, knot, ft/s
    ─────────────────────────────────────────────────────────────────────────

    Args:
        value: The numeric value to convert.
        from_unit: The source unit string (see list above, case-sensitive).
        to_unit:   The target unit string (see list above, case-sensitive).
        category:  Unit category — one of "length", "weight", "temperature", "speed".

    Returns:
        dict with keys:
          - value (float): Original value
          - from_unit (str): Source unit
          - to_unit (str): Target unit
          - category (str): Unit category
          - result (float): Converted value
          - error (str | None): Error message on failure, else null
    """
    # All factors are relative to a canonical SI base unit
    LENGTH_TO_M  = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3,
                    "inch": 0.0254, "ft": 0.3048, "yd": 0.9144,
                    "mile": 1609.344, "nautical_mile": 1852.0}
    WEIGHT_TO_KG = {"mg": 1e-6, "g": 1e-3, "kg": 1.0, "tonne": 1e3,
                    "oz": 0.028349523125, "lb": 0.45359237, "stone": 6.35029318}
    SPEED_TO_MS  = {"m/s": 1.0, "km/h": 1/3.6, "mph": 0.44704,
                    "knot": 0.514444, "ft/s": 0.3048}

    try:
        if category == "temperature":
            # Temperature requires special non-linear handling
            conversions = {
                ("C", "F"): lambda v: v * 9/5 + 32,
                ("C", "K"): lambda v: v + 273.15,
                ("F", "C"): lambda v: (v - 32) * 5/9,
                ("F", "K"): lambda v: (v - 32) * 5/9 + 273.15,
                ("K", "C"): lambda v: v - 273.15,
                ("K", "F"): lambda v: (v - 273.15) * 9/5 + 32,
            }
            if from_unit == to_unit:
                result = value
            elif (from_unit, to_unit) not in conversions:
                raise ValueError(
                    f"Cannot convert '{from_unit}' → '{to_unit}'. "
                    f"Supported temperature units: C, F, K."
                )
            else:
                result = conversions[(from_unit, to_unit)](value)

        else:
            table = {"length": LENGTH_TO_M, "weight": WEIGHT_TO_KG, "speed": SPEED_TO_MS}.get(category)
            if table is None:
                raise ValueError(f"Unknown category '{category}'. "
                                 "Choose from: length, weight, temperature, speed.")
            if from_unit not in table:
                raise ValueError(f"Unknown {category} unit '{from_unit}'. "
                                 f"Supported: {', '.join(sorted(table))}.")
            if to_unit not in table:
                raise ValueError(f"Unknown {category} unit '{to_unit}'. "
                                 f"Supported: {', '.join(sorted(table))}.")
            # Convert to SI base, then to target
            result = value * table[from_unit] / table[to_unit]

        return {
            "value": value, "from_unit": from_unit,
            "to_unit": to_unit, "category": category,
            "result": _pretty(result), "error": None,
        }
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return {
            "value": value, "from_unit": from_unit,
            "to_unit": to_unit, "category": category,
            "result": None, "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Tool registry — import this in your LangGraph agent
# ---------------------------------------------------------------------------

TOOLS = [
    calculate,
    basic_arithmetic,
    power_and_roots,
    percentage,
    gcd_lcm,
    prime_check,
    factorial_and_combo,
    statistics_basic,
    geometry,
    unit_convert,
]
