"""Scaled variant of Alart Curnier's normal projection."""

from functools import partial

import numpy as np
import porepy as pp
from abc import abstractmethod
import ncp


class ScaledAlartCurnier_NormalContact:
    normal_component: pp.ad.Operator
    """Normal component."""

    contact_traction: pp.ad.Operator
    """Contact traction."""

    displacement_jump: pp.ad.Operator
    """Displacement jump."""

    fracture_gap: pp.ad.Operator
    """Fracture gap."""

    contact_mechanics_numerical_constant: pp.ad.Operator
    """Augmentation constant."""

    numerical: pp.NumericalConstants
    """Numerical parameters."""

    @abstractmethod
    def scaling_exponent_normal_return(self, subdomains) -> pp.ad.Operator:
        """Scaling for the normal return projection.

        Exponent 0.0 corresponds to the normal return projection.
        Exponent 1.0 corresponds to the normal extension of the Hueber formulation.

        """
        raise NotImplementedError(
            "Method 'scaling_normal_return' must be implemented in the subclass."
        )

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Equation for the normal component of the contact mechanics.

        This constraint equation enforces non-penetration of opposing fracture
        interfaces. The equation is dimensionless, as we use nondimensionalized
        contact traction.

        Parameters:
            aubdomains: List of subdomains where the contact mechanics equation is
            defined.

        Returns:
            Operator for the normal deformation equation.

        """
        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        gap: pp.ad.Operator = self.fracture_gap(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        t_n_trial = t_n + c_num * (u_n - gap)

        # Maximum/cut-off function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")

        # Auxiliary functions.
        f_abs = pp.ad.Function(partial(pp.ad.l2_norm, 1), "abs_function")
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_isclose_times_identity = pp.ad.Function(
            partial(
                ncp.isclose_times_identity,
                self.numerical.open_state_tolerance,
                0,
            ),
            "isclose_times_identity_function",
        )
        f_gt_times_identity = pp.ad.Function(
            partial(
                ncp.gt_times_identity,
                self.numerical.open_state_tolerance,
            ),
            "greater_than_characteristic_times_identity_function",
        )

        # The complimentarity condition as scaled projection.
        equation: pp.ad.Operator = f_isclose_times_identity(
            f_abs(t_n_trial), t_n
        ) + f_gt_times_identity(
            f_abs(t_n_trial),
            f_abs(t_n_trial) ** self.scaling_exponent_normal_return(subdomains)
            * (t_n + f_max(-t_n_trial, zeros_frac)),
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation


class ConstantScaledAlartCurnier_NormalContact(ScaledAlartCurnier_NormalContact):
    def scaling_exponent_normal_return(self, subdomains) -> pp.ad.Operator:
        """Scaling for the normal return projection."""
        exponent = (
            0.1  # pp.ad.Scalar(self.params["contact"]["normal_scaling_exponent"])
        )
        return exponent


class RandomScaledAlartCurnier_NormalContact(ScaledAlartCurnier_NormalContact):
    equation_system: pp.ad.EquationSystem
    """Equation system for the model."""

    def before_nonlinear_iteration(self) -> None:
        if not hasattr(self, "random_scaling_exponent_alart_curnier_normal"):
            self.random_scaling_exponent_alart_curnier_normal = pp.ad.Scalar(0.0)
        if self.nonlinear_solver_statistics.num_iterations == 0:
            random_value = 0.0
        else:
            # random_value = np.clip(np.abs(np.random.normal(0, 0.33)), None, 1.0)
            rng = np.random.default_rng()
            random_value = rng.uniform(0, 1.0)
        self.random_scaling_exponent_alart_curnier_normal.set_value(random_value)

    def solver_info(self) -> dict[str, float]:
        """Return solver info for logging."""
        return {
            "random_scaling_exponent_alart_curnier_normal": (
                self.equation_system.evaluate(
                    self.random_scaling_exponent_alart_curnier_normal
                )
            )
        }

    def scaling_exponent_normal_return(self, subdomains) -> pp.ad.Operator:
        """Scaling for the radial return projection."""
        if not hasattr(self, "random_scaling_exponent_alart_curnier_normal"):
            self.random_scaling_exponent_alart_curnier_normal = pp.ad.Scalar(0.0)
        return self.random_scaling_exponent_alart_curnier_normal


class DecayingScaledAlartCurnier_NormalContact(ScaledAlartCurnier_NormalContact):
    def scaling_exponent_radial_return(self, subdomains) -> pp.ad.Operator:
        # Trial traction.
        nd_vec_to_normal = self.normal_component(subdomains)
        u_n = nd_vec_to_normal @ self.displacement_jump(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        gap: pp.ad.Operator = self.fracture_gap(subdomains)
        t_n_trial = t_n + c_num * (u_n - gap)

        # Positive part of trial traction.
        num_cells = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        t_n_trial_positive = f_max(t_n_trial, zeros_frac)

        # Exponent, being 1.0 inside the feasible regime, and decaying to 0.0 outside.
        f_exp = pp.ad.Function(pp.ad.exp, "exp_function")
        exponent = f_exp(-(t_n_trial_positive * t_n_trial_positive))
        return exponent
