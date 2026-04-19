"""Basic Python learning examples."""


def classify_number(value: int) -> str:
    """Return whether a number is positive, negative, or zero."""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def is_even(value: int) -> bool:
    """Return True when value is an even integer."""
    return value % 2 == 0

