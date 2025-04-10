"""NCP functions."""

import porepy as pp
from functools import partial
import numpy as np


def min(a: pp.ad.Operator, b: pp.ad.Operator, mu: float = 0.0) -> pp.ad.Operator:
    """Min function."""
    assert np.isclose(mu, 0.0), "mu not implemented"
    # regularized min-NCP: min(a,b) = 0.5 * (a+b - ((a-b)^2)^0.5
    # equation: pp.ad.Operator = pp.ad.Scalar(0.5) * (
    #    force + gap - ((force - gap) ** 2 + mu) ** 0.5
    # )
    f_max = pp.ad.Function(pp.ad.maximum, "max_function")
    return pp.ad.Scalar(-1.0) * f_max(pp.ad.Scalar(-1.0) * a, pp.ad.Scalar(-1.0) * b)


def fb(a: pp.ad.Operator, b: pp.ad.Operator, mu: float = 0.0) -> pp.ad.Operator:
    """Fischer-Burmeister function."""
    assert np.isclose(mu, 0.0), "mu not implemented"
    # Fischer-Burmeister: (a**2 + b**2)**0.5 - (a + b)
    # equation: pp.ad.Operator = (force + gap) - (force**2 + gap**2 + mu) ** 0.5
    return pp.ad.Scalar(0.5) * ((a + b) - (a**2 + b**2) ** 0.5)


def min_regularized_fb(
    a: pp.ad.Operator, b: pp.ad.Operator, tol: float = 1e-10
) -> pp.ad.Operator:
    """Fischer-Burmeister function regularized by min function."""
    f_characteristic_fb = pp.ad.Function(
        partial(pp.ad.functions.characteristic_function, tol),
        "characteristic_function_for_zero_normal_traction",
    )
    char_val = f_characteristic_fb(a**2 + b**2)
    min_ncp_equation: pp.ad.Operator = min(a, b)
    fb_ncp_equation = fb(a, b)
    return (
        char_val * min_ncp_equation + (pp.ad.Scalar(1.0) - char_val) * fb_ncp_equation
    )
