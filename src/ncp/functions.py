"""NCP functions."""

import porepy as pp
from functools import partial


def min(a: pp.ad.Operator, b: pp.ad.Operator, mu: float = 0.0) -> pp.ad.Operator:
    """Min function.

    Args:
        a: First operand.
        b: Second operand.
        mu: Regularization parameter.

    Returns:
        pp.ad.Operator: The minimum of a and b, regularized by mu.

    """
    return pp.ad.Scalar(0.5) * ((a + b) - ((a - b) ** 2 + pp.ad.Scalar(mu)) ** 0.5)


def fb(a: pp.ad.Operator, b: pp.ad.Operator, mu: float = 0.0) -> pp.ad.Operator:
    """Fischer-Burmeister function.

    Args:
        a: First operand.
        b: Second operand.
        mu: Regularization parameter.

    Returns:
        pp.ad.Operator: The Fischer-Burmeister function value.

    """
    return (pp.ad.Scalar(mu) + a**2 + b**2) ** 0.5 - (a + b)


def min_regularized_fb(
    a: pp.ad.Operator, b: pp.ad.Operator, tol: float = 1e-10, mu: float = 0.0
) -> pp.ad.Operator:
    """Fischer-Burmeister function, but in the origin regularized by min.

    Args:
        a: First operand.
        b: Second operand.
        tol: Tolerance for the regularization.
        mu: Regularization parameter.

    Returns:
        pp.ad.Operator: The regularized Fischer-Burmeister function value.

    """
    f_characteristic_fb = pp.ad.Function(
        partial(pp.ad.functions.characteristic_function, tol),
        "characteristic_function_for_zero_normal_traction",
    )
    char_val = f_characteristic_fb(a**2 + b**2)
    return char_val * min(a, b, mu) + (pp.ad.Scalar(1.0) - char_val) * fb(a, b, mu)
