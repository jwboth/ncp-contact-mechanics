"""Utilities like AD functions."""

from typing import TypeVar
import numpy as np
import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray

FloatType = TypeVar("FloatType", AdArray, np.ndarray, float)


def nan_to_num(var: FloatType) -> FloatType:
    if isinstance(var, AdArray):
        val = np.nan_to_num(var.val)
        jac = var._diagvec_mul_jac(np.zeros_like(var.val))
        return AdArray(val, 0 * jac)
    else:
        return np.nan_to_num(var)


def sign(var: FloatType) -> FloatType:
    tol = -1e-12
    if isinstance(var, AdArray):
        val = np.ones_like(var.val, dtype=var.val.dtype)
        neg_inds = var.val < tol
        val[neg_inds] = -1
        jac = var._diagvec_mul_jac(np.sign(var.val))
        return AdArray(val, 0 * jac)
    else:
        return np.sign(var)


def log_reg(var: FloatType) -> FloatType:
    if isinstance(var, AdArray):
        mask = np.abs(var.val) < 1
        val = var.val
        val[~mask] = np.log(np.abs(var.val[~mask])) + 1
        jac = var._diagvec_mul_jac(np.ones_like(var.val))
        jac_outer = np.ones_like(var.val)
        jac_outer[~mask] = 1.0 / np.abs(var.val[~mask])
        jac = var._diagvec_mul_jac(jac_outer)
        return AdArray(val, jac)
    elif isinstance(var, np.ndarray):
        mask = np.abs(var) < 1
        val = var.copy()
        val[~mask] = np.log(np.abs(var[~mask])) + 1
        return val
    else:
        assert False


def abs_reg(var: FloatType) -> FloatType:
    if isinstance(var, AdArray):
        val = np.abs(var.val)
        sign = np.ones_like(val, dtype=val.dtype)
        neg_inds = var.val < -1e-12
        sign[neg_inds] = -1
        jac = var._diagvec_mul_jac(sign)
        return AdArray(val, jac)
    else:
        return np.abs(var)


def l2_norm_reg(dim: int, var: pp.ad.AdArray) -> pp.ad.AdArray:
    """L2 norm of a vector variable.

    For the example of dim=3 components and n vectors, the ordering is assumed
    to be ``[u0, v0, w0, u1, v1, w1, ..., un, vn, wn]``.

    Vectors satisfying ui=vi=wi=0 are assigned positive entries in the Jacobi
    matrix.

    Note:
        See module level documentation on how to wrap functions like this in ad.Function.

    Parameters:
        dim: Dimension, i.e. number of vector components.
        var: Ad operator which is argument of the norm function.

    Returns:
        The norm of var with appropriate val and jac attributes.

    """
    if not isinstance(var, AdArray):
        resh = np.reshape(var, (dim, -1), order="F")
        return np.linalg.norm(resh, axis=0)
    if dim == 1:
        # For scalar variables, the cell-wise L2 norm is equivalent to
        # taking the absolute value.
        return pp.ad.functions.abs_reg(var)
    resh = np.reshape(var.val, (dim, -1), order="F")
    vals = np.linalg.norm(resh, axis=0)
    # Avoid dividing by zero
    tol = 1e-12
    nonzero_inds = vals > tol
    jac_vals = np.ones(resh.shape)
    jac_vals[:, nonzero_inds] = resh[:, nonzero_inds] / vals[nonzero_inds]
    # Prepare for left multiplication with var.jac to yield
    # norm(var).jac = var/norm(var) * var.jac
    dim_size = var.val.size
    # Check that size of var is compatible with the given dimension, e.g. all 'cells'
    # have the same number of values assigned
    assert dim_size % dim == 0
    size = int(dim_size / dim)
    local_inds_t = np.arange(dim_size)
    if size == 0:
        local_inds_n = np.empty(0, dtype=np.int32)
    else:
        local_inds_n = np.array(np.kron(np.arange(size), np.ones(dim)), dtype=np.int32)
    norm_jac = sps.csr_matrix(
        (jac_vals.ravel("F"), (local_inds_n, local_inds_t)),
        shape=(size, dim_size),
    )
    jac = norm_jac * var.jac
    return pp.ad.AdArray(vals, jac)
