"""Scaled variant of Alart Curnier's normal projection."""

from functools import partial

import numpy as np
import porepy as pp
from abc import abstractmethod


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
            subdomains: List of subdomains where the contact mechanics equation is
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
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")

        # Projection
        projection_n = max_function(-t_n_trial, zeros_frac)

        # Scaling
        f_abs = pp.ad.Function(partial(pp.ad.l2_norm, 1), "abs_function")
        scaling = projection_n ** self.scaling_exponent_normal_return(subdomains)

        # Characteristic function for the origin
        f_characteristic = pp.ad.Function(
            partial(
                pp.ad.functions.characteristic_function,
                self.numerical.open_state_tolerance,
            ),
            "characteristic_function_for_zero_normal_traction",
        )
        characteristic_origin = f_characteristic(f_abs(t_n_trial))
        characteristic_origin.set_name("characteristic_origin")
        characteristic_rest = pp.ad.Scalar(1.0) - characteristic_origin
        characteristic_rest.set_name("characteristic_rest")

        # The complimentarity condition as scaled projection
        # equation: pp.ad.Operator = t_n - min_function(t_n_trial, zeros_frac)
        equation: pp.ad.Operator = characteristic_origin * t_n + characteristic_rest * (
            scaling * (t_n + max_function(-t_n_trial, zeros_frac))
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation


class ConstantScaledAlartCurnier_NormalContact(ScaledAlartCurnier_NormalContact):
    def scaling_exponent_normal_return(self, subdomains) -> pp.ad.Operator:
        """Scaling for the normal return projection."""
        exponent = pp.ad.Scalar(self.params["contact"]["normal_scaling_exponent"])
        return exponent


class RandomScaledAlartCurnier_NormalContact(ScaledAlartCurnier_NormalContact):
    equation_system: pp.ad.EquationSystem
    """Equation system for the model."""

    def before_nonlinear_iteration(self) -> None:
        if not hasattr(self, "random_scaling_exponent_alart_curnier_normal"):
            self.random_scaling_exponent_alart_curnier_normal = pp.ad.Scalar(1.0)

        self.random_scaling_exponent_alart_curnier_normal.set_value(
            np.clip(np.random.normal(0, 1) ** 2, None, 1.0)
        )

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
            self.random_scaling_exponent_alart_curnier_normal = pp.ad.Scalar(
                np.clip(np.random.normal(0, 1) ** 2, None, 1.0)
            )

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
