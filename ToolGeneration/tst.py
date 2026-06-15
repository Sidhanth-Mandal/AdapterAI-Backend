tool = {
  "tool_name": "probability_expert_tools",
  "tool_description": "A suite of tools for a conversational probability expert AI assistant. Includes random integer generation in exclusive range (1,100), structured probability calculation with LaTeX output, and safe mathematical expression to LaTeX conversion using sympy.",
  "category": "mathematics",
  "dependencies": [
    "sympy"
  ],
  "functions": [
    {
      "name": "generate_random_int",
      "description": "Generate a uniformly random integer in the exclusive range (1, 100), i.e., between 2 and 99 inclusive. Supports optional seed for reproducibility. Returns the generated value and the Python code snippet used.",
      "parameters": [
        {
          "name": "lower",
          "type": "int",
          "description": "Inclusive lower bound (fixed default: 2)",
          "required": False,
          "example": 2
        },
        {
          "name": "upper",
          "type": "int",
          "description": "Inclusive upper bound (fixed default: 99)",
          "required": False,
          "example": 99
        },
        {
          "name": "seed",
          "type": "int",
          "description": "Optional seed for reproducibility",
          "required": False,
          "example": 42
        }
      ],
      "outputs": [
        {
          "name": "value",
          "type": "int",
          "description": "The generated random integer",
          "example": 47
        },
        {
          "name": "code_snippet",
          "type": "str",
          "description": "The exact Python line used to generate the value",
          "example": "random.randint(2, 99)"
        },
        {
          "name": "error",
          "type": "str",
          "description": "Error message if generation failed",
          "example": "lower must be strictly less than upper"
        }
      ],
      "return_type": "dict"
    },
    {
      "name": "calculate_probability",
      "description": "Evaluate simple probability expressions using structured event descriptions or explicit numerator/denominator. Returns decimal probability, reduced fraction string, and inline LaTeX. Supports standard deck, coin, and dice lookups.",
      "parameters": [
        {
          "name": "event",
          "type": "dict",
          "description": "Description of the probability event, e.g. {\"type\": \"draw\", \"deck\": \"standard\", \"cards\": [\"A\"]} or {\"type\": \"coin\", \"flips\": 3, \"heads\": 2} or {\"type\": \"dice\", \"sides\": 6, \"target\": 4}",
          "required": True,
          "example": {"type": "draw", "deck": "standard", "cards": ["A"]}
        },
        {
          "name": "total_outcomes",
          "type": "int",
          "description": "Optional explicit denominator (total outcomes)",
          "required": False,
          "example": 52
        },
        {
          "name": "favorable_outcomes",
          "type": "int",
          "description": "Optional explicit numerator (favorable outcomes)",
          "required": False,
          "example": 4
        }
      ],
      "outputs": [
        {
          "name": "probability",
          "type": "float",
          "description": "Decimal probability value between 0 and 1",
          "example": 0.07692307692307693
        },
        {
          "name": "fraction",
          "type": "str",
          "description": "Reduced fraction as a string",
          "example": "1/13"
        },
        {
          "name": "latex",
          "type": "str",
          "description": "Inline LaTeX representation of the reduced fraction",
          "example": "\\frac{1}{13}"
        },
        {
          "name": "explanation",
          "type": "str",
          "description": "Human-readable explanation of the calculation",
          "example": "Drawing an Ace from a standard 52-card deck: 4 favorable out of 52 total."
        },
        {
          "name": "error",
          "type": "str",
          "description": "Error message if calculation failed",
          "example": "Unsupported event type: 'roulette'"
        }
      ],
      "return_type": "dict"
    },
    {
      "name": "format_latex",
      "description": "Convert a plain-text mathematical expression into a safe inline LaTeX string using sympy. Only allows whitelisted operators and alphanumeric symbols. Returns the LaTeX representation.",
      "parameters": [
        {
          "name": "expression",
          "type": "str",
          "description": "Plain-text math expression to convert, e.g. 'a/b' or 'C(5,2)/2**5'",
          "required": True,
          "example": "4/52"
        }
      ],
      "outputs": [
        {
          "name": "latex",
          "type": "str",
          "description": "Inline LaTeX string representation of the expression",
          "example": "\\frac{1}{13}"
        },
        {
          "name": "error",
          "type": "str",
          "description": "Error message if conversion failed",
          "example": "Disallowed characters detected in expression"
        }
      ],
      "return_type": "dict"
    }
  ],
  "code": "import random\nimport math\nimport fractions\nimport logging\nimport os\nimport signal\nimport threading\nfrom functools import lru_cache\nfrom typing import Any, Dict, Optional\n\n# ---------------------------------------------------------------------------\n# Logging\n# ---------------------------------------------------------------------------\nlogger = logging.getLogger(\"probability_expert_tools\")\nif not logger.handlers:\n    _handler = logging.StreamHandler()\n    _handler.setFormatter(logging.Formatter(\"%(levelname)s | %(name)s | %(message)s\"))\n    logger.addHandler(_handler)\nlogger.setLevel(logging.INFO)\n\n\n# ---------------------------------------------------------------------------\n# Timeout helper (cross-platform)\n# ---------------------------------------------------------------------------\nclass _TimeoutError(Exception):\n    \"\"\"Raised when a function exceeds its allowed wall-clock time.\"\"\"\n\n\ndef _run_with_timeout(fn, timeout_seconds: float, *args, **kwargs):\n    \"\"\"\n    Execute *fn* with positional *args* and keyword *kwargs*.\n    Raises _TimeoutError if execution takes longer than *timeout_seconds*.\n    Uses signal.alarm on Unix; falls back to a threading.Timer on Windows.\n    \"\"\"\n    result_container: Dict[str, Any] = {}\n\n    # --- POSIX / Unix path ---\n    if hasattr(signal, \"SIGALRM\"):\n        def _handler(signum, frame):  # noqa: ANN001\n            raise _TimeoutError(f\"Function exceeded {timeout_seconds}s timeout\")\n\n        old_handler = signal.signal(signal.SIGALRM, _handler)\n        # alarm only accepts integer seconds; use math.ceil for safety\n        signal.alarm(max(1, math.ceil(timeout_seconds)))\n        try:\n            result_container[\"value\"] = fn(*args, **kwargs)\n        finally:\n            signal.alarm(0)\n            signal.signal(signal.SIGALRM, old_handler)\n        return result_container[\"value\"]\n\n    # --- Windows / threading fallback ---\n    exc_container: Dict[str, Any] = {}\n\n    def _target():\n        try:\n            result_container[\"value\"] = fn(*args, **kwargs)\n        except Exception as exc:  # noqa: BLE001\n            exc_container[\"exc\"] = exc\n\n    thread = threading.Thread(target=_target, daemon=True)\n    thread.start()\n    thread.join(timeout=timeout_seconds)\n    if thread.is_alive():\n        raise _TimeoutError(f\"Function exceeded {timeout_seconds}s timeout\")\n    if \"exc\" in exc_container:\n        raise exc_container[\"exc\"]\n    return result_container.get(\"value\")\n\n\n# ---------------------------------------------------------------------------\n# Retry decorator template (for future network-bound extensions)\n# ---------------------------------------------------------------------------\ndef _retry(max_retries: int = 3, base: float = 0.2, factor: float = 2.0):\n    \"\"\"\n    Exponential-backoff retry decorator.\n    Parameters\n    ----------\n    max_retries : int\n        Maximum number of attempts (including the first).\n    base : float\n        Initial wait in seconds before the first retry.\n    factor : float\n        Multiplier applied to the wait after each failure.\n    \"\"\"\n    import time\n    import functools\n\n    def decorator(fn):\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            wait = base\n            last_exc: Optional[Exception] = None\n            for attempt in range(max_retries):\n                try:\n                    return fn(*args, **kwargs)\n                except Exception as exc:  # noqa: BLE001\n                    last_exc = exc\n                    logger.warning(\n                        \"Attempt %d/%d failed for %s: %s\",\n                        attempt + 1,\n                        max_retries,\n                        fn.__name__,\n                        exc,\n                    )\n                    if attempt < max_retries - 1:\n                        time.sleep(wait)\n                        wait *= factor\n            raise last_exc  # type: ignore[misc]\n\n        return wrapper\n\n    return decorator\n\n\n# ===========================================================================\n# TOOL 1 — generate_random_int\n# ===========================================================================\n\ndef generate_random_int(\n    lower: int = 2,\n    upper: int = 99,\n    seed: Optional[int] = None,\n) -> Dict[str, Any]:\n    \"\"\"\n    Generate a uniformly random integer in the inclusive range [lower, upper],\n    which corresponds to the exclusive range (1, 100) by default.\n\n    Parameters\n    ----------\n    lower : int\n        Inclusive lower bound.  Default ``2``.\n    upper : int\n        Inclusive upper bound.  Default ``99``.\n    seed  : int, optional\n        If supplied, the PRNG is seeded for reproducibility.\n\n    Returns\n    -------\n    dict\n        ::\n\n            {\n                \"value\": int,             # the generated integer\n                \"code_snippet\": str       # Python line that produced it\n            }\n\n        On error::\n\n            {\"error\": \"<human-readable message>\"}\n\n    Examples\n    --------\n    >>> result = generate_random_int(seed=42)\n    >>> assert 2 <= result[\"value\"] <= 99\n    \"\"\"\n    # --- Input validation ---\n    if not isinstance(lower, int) or isinstance(lower, bool):\n        return {\"error\": \"'lower' must be a plain integer.\"}\n    if not isinstance(upper, int) or isinstance(upper, bool):\n        return {\"error\": \"'upper' must be a plain integer.\"}\n    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):\n        return {\"error\": \"'seed' must be an integer or None.\"}\n    if lower >= upper:\n        return {\n            \"error\": (\n                f\"'lower' ({lower}) must be strictly less than 'upper' ({upper}).\"\n            )\n        }\n\n    def _generate() -> Dict[str, Any]:\n        rng = random.Random(seed)\n        value = rng.randint(lower, upper)\n        snippet = f\"random.randint({lower}, {upper})\"\n        return {\"value\": value, \"code_snippet\": snippet}\n\n    try:\n        return _run_with_timeout(_generate, timeout_seconds=0.1)\n    except _TimeoutError as exc:\n        logger.error(\"generate_random_int timed out: %s\", exc)\n        return {\"error\": str(exc)}\n    except Exception as exc:  # noqa: BLE001\n        logger.error(\"generate_random_int unexpected error: %s\", exc)\n        return {\"error\": f\"Unexpected error: {exc}\"}\n\n\n# ===========================================================================\n# TOOL 2 — calculate_probability\n# ===========================================================================\n\n@lru_cache(maxsize=1)\ndef _standard_deck_counts() -> Dict[str, int]:\n    \"\"\"\n    Return a mapping of card identifiers to their count in a standard\n    52-card deck.  Result is cached after the first call.\n    \"\"\"\n    ranks = [\"A\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\", \"J\", \"Q\", \"K\"]\n    suits = [\"hearts\", \"diamonds\", \"clubs\", \"spades\"]\n    counts: Dict[str, int] = {}\n    for rank in ranks:\n        counts[rank] = 4  # 4 suits per rank\n    for suit in suits:\n        counts[suit] = 13  # 13 ranks per suit\n    counts[\"red\"] = 26  # hearts + diamonds\n    counts[\"black\"] = 26  # clubs + spades\n    counts[\"face\"] = 12  # J, Q, K across 4 suits\n    counts[\"number\"] = 40  # A treated as rank here; 2-10 = 9 ranks × 4\n    return counts\n\n\ndef _fraction_to_latex(frac: fractions.Fraction) -> str:\n    \"\"\"Convert a :class:`fractions.Fraction` to an inline LaTeX string.\"\"\"\n    if frac.denominator == 1:\n        return str(frac.numerator)\n    return rf\"\\frac{{{frac.numerator}}}{{{frac.denominator}}}\"\n\n\ndef _resolve_event(event: Dict[str, Any]) -> Dict[str, Any]:\n    \"\"\"\n    Resolve a structured event dict into favorable and total outcome counts.\n\n    Supported event types\n    ---------------------\n    ``draw``\n        Drawing cards from a standard 52-card deck.\n        ``event[\"cards\"]`` is a list of rank/suit/colour identifiers.\n\n    ``coin``\n        Flipping *n* fair coins and getting exactly *k* heads.\n        ``event[\"flips\"]`` = n, ``event[\"heads\"]`` = k.\n\n    ``dice``\n        Rolling a fair *s*-sided die and landing on ``event[\"target\"]``\n        (or any of ``event[\"targets\"]``).\n\n    Returns\n    -------\n    dict\n        ``{\"favorable\": int, \"total\": int, \"explanation\": str}``\n        or ``{\"error\": str}``.\n    \"\"\"\n    event_type = str(event.get(\"type\", \"\")).lower()\n\n    # ------------------------------------------------------------------ draw\n    if event_type == \"draw\":\n        deck_type = str(event.get(\"deck\", \"standard\")).lower()\n        if deck_type != \"standard\":\n            return {\"error\": f\"Unsupported deck type: '{deck_type}'. Only 'standard' is supported.\"}\n\n        cards = event.get(\"cards\", [])\n        if not isinstance(cards, list) or len(cards) == 0:\n            return {\"error\": \"'cards' must be a non-empty list of rank/suit identifiers.\"}\n\n        deck_counts = _standard_deck_counts()\n        total = 52\n        favorable = 0\n        labels = []\n        for card in cards:\n            card_key = str(card).strip()\n            if card_key not in deck_counts:\n                return {\"error\": f\"Unknown card identifier: '{card_key}'.\"}\n            favorable += deck_counts[card_key]\n            labels.append(card_key)\n\n        explanation = (\n            f\"Drawing {' or '.join(labels)} from a standard 52-card deck: \"\n            f\"{favorable} favorable out of {total} total.\"\n        )\n        return {\"favorable\": favorable, \"total\": total, \"explanation\": explanation}\n\n    # ------------------------------------------------------------------ coin\n    if event_type == \"coin\":\n        n = event.get(\"flips\")\n        k = event.get(\"heads\")\n        if not isinstance(n, int) or not isinstance(k, int):\n            return {\"error\": \"'flips' and 'heads' must be integers for coin events.\"}\n        if k < 0 or k > n:\n            return {\"error\": f\"'heads' ({k}) must be between 0 and 'flips' ({n}).\"}\n\n        total = 2 ** n\n        favorable = math.comb(n, k)\n        explanation = (\n            f\"Probability of exactly {k} heads in {n} fair coin flip(s): \"\n            f\"C({n},{k}) = {favorable} favorable out of {total} total.\"\n        )\n        return {\"favorable\": favorable, \"total\": total, \"explanation\": explanation}\n\n    # ------------------------------------------------------------------ dice\n    if event_type == \"dice\":\n        sides = event.get(\"sides\", 6)\n        if not isinstance(sides, int) or sides < 2:\n            return {\"error\": \"'sides' must be an integer >= 2 for dice events.\"}\n\n        # Accept either a single target or a list of targets\n        if \"targets\" in event:\n            targets = event[\"targets\"]\n            if not isinstance(targets, list):\n                return {\"error\": \"'targets' must be a list of integers.\"}\n        elif \"target\" in event:\n            targets = [event[\"target\"]]\n        else:\n            return {\"error\": \"Dice events require 'target' (int) or 'targets' (list of int).\"}\n\n        valid_targets = [t for t in targets if isinstance(t, int) and 1 <= t <= sides]\n        if len(valid_targets) != len(targets):\n            return {\n                \"error\": (\n                    f\"All targets must be integers between 1 and {sides}.\"\n                )\n            }\n\n        total = sides\n        favorable = len(valid_targets)\n        target_str = \", \".join(str(t) for t in valid_targets)\n        explanation = (\n            f\"Rolling a {sides}-sided die and landing on {target_str}: \"\n            f\"{favorable} favorable out of {total} total.\"\n        )\n        return {\"favorable\": favorable, \"total\": total, \"explanation\": explanation}\n\n    return {\"error\": f\"Unsupported event type: '{event_type}'. Supported: 'draw', 'coin', 'dice'.\"}\n\n\ndef calculate_probability(\n    event: Dict[str, Any],\n    total_outcomes: Optional[int] = None,\n    favorable_outcomes: Optional[int] = None,\n) -> Dict[str, Any]:\n    \"\"\"\n    Compute the probability of a structured event and return the result\n    as a decimal, reduced fraction, and inline LaTeX.\n\n    Parameters\n    ----------\n    event : dict\n        Structured description of the probability event.  Required keys\n        depend on the event type:\n\n        * ``{\"type\": \"draw\", \"deck\": \"standard\", \"cards\": [\"A\"]}``\n        * ``{\"type\": \"coin\", \"flips\": 3, \"heads\": 2}``\n        * ``{\"type\": \"dice\", \"sides\": 6, \"target\": 4}``\n\n    total_outcomes : int, optional\n        Explicit denominator.  If *both* ``total_outcomes`` and\n        ``favorable_outcomes`` are provided the event lookup is skipped.\n    favorable_outcomes : int, optional\n        Explicit numerator.\n\n    Returns\n    -------\n    dict\n        ::\n\n            {\n                \"probability\": float,\n                \"fraction\": \"a/b\",\n                \"latex\": \"\\\\frac{a}{b}\",\n                \"explanation\": str\n            }\n\n        On error::\n\n            {\"error\": \"<human-readable message>\"}\n\n    Examples\n    --------\n    >>> r = calculate_probability({\"type\": \"draw\", \"deck\": \"standard\", \"cards\": [\"A\"]})\n    >>> r[\"fraction\"]\n    '1/13'\n    \"\"\"\n    # --- Input validation ---\n    if not isinstance(event, dict):\n        return {\"error\": \"'event' must be a dict.\"}\n\n    def _calculate() -> Dict[str, Any]:\n        nonlocal total_outcomes, favorable_outcomes\n\n        explanation = \"\"\n\n        # Fast path: both numerator and denominator supplied explicitly\n        if total_outcomes is not None and favorable_outcomes is not None:\n            if not isinstance(total_outcomes, int) or total_outcomes <= 0:\n                return {\"error\": \"'total_outcomes' must be a positive integer.\"}\n            if not isinstance(favorable_outcomes, int) or favorable_outcomes < 0:\n                return {\"error\": \"'favorable_outcomes' must be a non-negative integer.\"}\n            if favorable_outcomes > total_outcomes:\n                return {\n                    \"error\": (\n                        \"'favorable_outcomes' cannot exceed 'total_outcomes'.\"\n                    )\n                }\n            explanation = (\n                f\"{favorable_outcomes} favorable outcomes out of \"\n                f\"{total_outcomes} total.\"\n            )\n        else:\n            resolved = _resolve_event(event)\n            if \"error\" in resolved:\n                return resolved\n            favorable_outcomes = resolved[\"favorable\"]\n            total_outcomes = resolved[\"total\"]\n            explanation = resolved.get(\"explanation\", \"\")\n\n        frac = fractions.Fraction(favorable_outcomes, total_outcomes)\n        probability = float(frac)\n        fraction_str = f\"{frac.numerator}/{frac.denominator}\"\n        latex_str = _fraction_to_latex(frac)\n\n        return {\n            \"probability\": probability,\n            \"fraction\": fraction_str,\n            \"latex\": latex_str,\n            \"explanation\": explanation,\n        }\n\n    try:\n        return _run_with_timeout(_calculate, timeout_seconds=0.2)\n    except _TimeoutError as exc:\n        logger.error(\"calculate_probability timed out: %s\", exc)\n        return {\"error\": str(exc)}\n    except Exception as exc:  # noqa: BLE001\n        logger.error(\"calculate_probability unexpected error: %s\", exc)\n        return {\"error\": f\"Unexpected error: {exc}\"}\n\n\n# ===========================================================================\n# TOOL 3 — format_latex\n# ===========================================================================\n\n# Whitelist of characters allowed in expressions passed to format_latex.\n_ALLOWED_CHARS = set(\n    \"abcdefghijklmnopqrstuvwxyz\"\n    \"ABCDEFGHIJKLMNOPQRSTUVWXYZ\"\n    \"0123456789\"\n    \" +-*/^().,_\"\n)\n\n\ndef format_latex(expression: str) -> Dict[str, Any]:\n    \"\"\"\n    Convert a plain-text mathematical expression to an inline LaTeX string\n    using :func:`sympy.sympify` and :func:`sympy.latex`.\n\n    Only characters in the whitelist\n    ``[A-Za-z0-9 +-*/^().,_]`` are accepted to prevent injection.\n\n    Parameters\n    ----------\n    expression : str\n        Plain-text expression, e.g. ``\"4/52\"`` or ``\"C(5,2)/2**5\"``.\n\n    Returns\n    -------\n    dict\n        ::\n\n            {\"latex\": \"<inline LaTeX string>\"}\n\n        On error::\n\n            {\"error\": \"<human-readable message>\"}\n\n    Examples\n    --------\n    >>> result = format_latex(\"4/52\")\n    >>> result[\"latex\"]\n    '\\\\frac{1}{13}'\n    \"\"\"\n    if not isinstance(expression, str) or not expression.strip():\n        return {\"error\": \"'expression' must be a non-empty string.\"}\n\n    disallowed = set(expression) - _ALLOWED_CHARS\n    if disallowed:\n        return {\n            \"error\": (\n                f\"Disallowed characters detected: \"\n                f\"{sorted(disallowed)!r}. \"\n                \"Only alphanumeric characters and '+ - * / ** ^ ( )' are allowed.\"\n            )\n        }\n\n    def _convert() -> Dict[str, Any]:\n        try:\n            import sympy  # noqa: PLC0415\n        except ImportError:\n            return {\"error\": \"sympy is not installed. Run: pip install sympy\"}\n\n        # Replace ^ with ** for sympy compatibility\n        safe_expr = expression.replace(\"^\", \"**\")\n\n        try:\n            parsed = sympy.sympify(safe_expr, evaluate=True)\n        except (sympy.SympifyError, SyntaxError, TypeError) as exc:\n            return {\"error\": f\"Failed to parse expression: {exc}\"}\n\n        try:\n            latex_str = sympy.latex(parsed)\n        except Exception as exc:  # noqa: BLE001\n            return {\"error\": f\"Failed to convert to LaTeX: {exc}\"}\n\n        return {\"latex\": latex_str}\n\n    try:\n        return _run_with_timeout(_convert, timeout_seconds=0.1)\n    except _TimeoutError as exc:\n        logger.error(\"format_latex timed out: %s\", exc)\n        return {\"error\": str(exc)}\n    except Exception as exc:  # noqa: BLE001\n        logger.error(\"format_latex unexpected error: %s\", exc)\n        return {\"error\": f\"Unexpected error: {exc}\"}\n\n\n# ===========================================================================\n# Quick self-test (only executed when the module is run directly)\n# ===========================================================================\nif __name__ == \"__main__\":\n    import json\n\n    print(\"=== generate_random_int ===\")\n    print(json.dumps(generate_random_int(seed=42), indent=2))\n\n    print(\"\\n=== calculate_probability: ace from deck ===\")\n    print(json.dumps(\n        calculate_probability({\"type\": \"draw\", \"deck\": \"standard\", \"cards\": [\"A\"]}),\n        indent=2,\n    ))\n\n    print(\"\\n=== calculate_probability: 2 heads in 3 coin flips ===\")\n    print(json.dumps(\n        calculate_probability({\"type\": \"coin\", \"flips\": 3, \"heads\": 2}),\n        indent=2,\n    ))\n\n    print(\"\\n=== calculate_probability: rolling a 6 on a d6 ===\")\n    print(json.dumps(\n        calculate_probability({\"type\": \"dice\", \"sides\": 6, \"target\": 6}),\n        indent=2,\n    ))\n\n    print(\"\\n=== calculate_probability: explicit numerator/denominator ===\")\n    print(json.dumps(\n        calculate_probability({\"type\": \"custom\"}, total_outcomes=36, favorable_outcomes=6),\n        indent=2,\n    ))\n\n    print(\"\\n=== format_latex ===\")\n    print(json.dumps(format_latex(\"4/52\"), indent=2))\n    print(json.dumps(format_latex(\"C(5,2)/2**5\"), indent=2))\n"
}


from .validator import validate_tool

print(validate_tool(tool).summary())