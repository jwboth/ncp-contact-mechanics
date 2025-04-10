"""Module collecting useful convergence criteria for integration in solution strategy.

These can also serve as inspiration for how to define custom criteria to override
methods for computing norms in solution_strategy. But moreover, they can be used
for convergence studies etc.

Example:
    # Given a model class `MyModel`, to equip it with a custom metric to be used in
    # the solution strategy, one can override the corresponding method as follows:

    class MyNewModel(MyModel):

        def compute_nonlinear_increment_norm(self, solution: np.ndarray) -> float:
            # Method for computing the norm of the nonlinear increment during
            # `check_convergence`.
            return pp.LebesgueMetric().variable_norm(self, solution)

"""

from functools import partial
from typing import Callable, Optional

import numpy as np

import porepy as pp


class EuclideanMetric:  # (BaseMetric):
    """Purely algebraic metric (blind to physics and dimension).

    Simple but fairly robust convergence criterion. More advanced options are
    e.g. considering errors for each variable and/or each grid separately,
    possibly using _l2_norm_cell

    We normalize by the size of the vector as proxy for domain size.

    """

    def norm(self, values: np.ndarray) -> float:
        """Implementation of Euclidean l2 norm of the full vector, scaled by vector size.

        Parameters:
            values: algebraic respresentation of a mixed-dimensional variable

        Returns:
            float: measure of values

        """
        return np.linalg.norm(values) / np.sqrt(values.size)


class LebesgueMetric:  # (BaseMetric):
    """Dimension-consistent Lebesgue metric (blind to physics), but separates dimensions."""

    equation_system: pp.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    volume_integral: Callable[[pp.ad.Operator, list[pp.Grid], int], pp.ad.Operator]
    """General volume integral operator, defined in `pp.BalanceEquation`."""

    def variable_norm(
        self,
        values: np.ndarray,
        variables: Optional[list[pp.ad.Variable]] = None,
        split: bool = False,
    ) -> float:
        """Implementation of mixed-dimensional Lebesgue L2 norm of a physical state.

        Parameters:
            values: algebraic respresentation of a mixed-dimensional variable
            variables: list of variables to be considered

        Returns:
            float: measure of values

        """
        # Initialize container for collecting separate L2 norms.
        integrals = []

        # Use the equation system to get a view onto mixed-dimensional data structures.
        # Compute the L2 norm of each variable separately, automatically taking into
        # account volume and specific volume
        if variables is None:
            variables = self.equation_system.variables
        for variable in variables:
            # Assume low-order discretization with 1 DOF per active entity
            variable_dim = variable._cells + variable._faces + variable._nodes
            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, variable_dim), "l2_norm")
            sd = variable.domain
            indices = self.equation_system.dofs_of([variable])
            ad_values = pp.ad.DenseArray(values[indices])
            integral_squarred = np.sum(
                self.volume_integral(l2_norm(ad_values) ** 2, [sd], 1).value(
                    self.equation_system
                )
            )

            # Collect the L2 norms.
            integrals.append(np.sqrt(integral_squarred))

        return np.array(integrals) if split else np.sqrt(np.sum(np.square(integrals)))

    def residual_norm(
        self,
        equations=None,
        split: bool = False,
    ) -> float:
        """Essentially an L1-norm as the residuals already integrate over cells."""

        equation_names = list(self.equation_system._parse_equations(equations).keys())
        residuals = []

        for name in equation_names:
            # this raises a key error if a given equation name is unknown
            eq = self.equation_system._equations[name]
            # Note equation is already integrated over cells
            res = np.linalg.norm(eq.value(self.equation_system), ord=1)
            residuals.append(res)

        return np.array(residuals) if split else np.sum(residuals)
