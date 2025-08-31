from functools import partial

import numpy as np
import porepy as pp
from abc import abstractmethod


class ScaledRadialReturnTangentialContact:
    @abstractmethod
    def scaling_exponent_radial_return(self, subdomains) -> pp.ad.Operator:
        """Scaling for the radial return projection.

        Exponent 0.0 corresponds to the radial return projection.
        Exponent 1.0 corresponds to the Hueber formulation.

        """
        raise NotImplementedError(
            "Method 'scaling_exponent_radial_return' must be implemented in the subclass."
        )

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints."""

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])

        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # To map a scalar to the tangential plane, we need to sum the basis vectors.
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Variables: The tangential component of the contact traction and the plastic
        # displacement jump, and its time increment.
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )

        # The time increment of the tangential displacement jump.
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Auxiliary functions.
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        f_characteristic = pp.ad.Function(
            partial(
                pp.ad.functions.characteristic_function,
                self.numerical.open_state_tolerance,
            ),
            "characteristic_function_for_zero_normal_traction",
        )

        # Augment the traction.
        c_num = scalar_to_tangential @ self.contact_mechanics_numerical_constant(
            subdomains
        )
        t_t_trial = t_t + c_num * u_t_increment
        t_t_trial.set_name("t_t_trial")

        norm_t_t_trial = f_norm(t_t_trial)
        norm_t_t_trial.set_name("norm_t_t_trial")

        # Determine characteristic function, when to apply projection.
        # Only not close to the origin.
        characteristic: pp.ad.Operator = pp.ad.Scalar(1.0) - f_characteristic(
            norm_t_t_trial
        )
        characteristic.set_name("characteristic")

        # Define scalings and integrate in the radial return projection.
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        scaling = f_max(norm_t_t_trial, b_p)  # order on purpose for slip
        t_t_scaling = scaling ** self.scaling_exponent_radial_return(subdomains)
        t_t_trial_scaling = b_p * scaling ** (
            self.scaling_exponent_radial_return(subdomains) - pp.ad.Scalar(1.0)
        )

        chi_open = f_characteristic(b_p)
        chi_closed = pp.ad.Scalar(1.0) - chi_open

        equation_open = t_t
        equation_closed = (scalar_to_tangential @ t_t_scaling) * t_t - (
            scalar_to_tangential @ t_t_trial_scaling
        ) * t_t_trial
        equation: pp.ad.Operator = (scalar_to_tangential @ chi_open) * equation_open + (
            scalar_to_tangential @ chi_closed
        ) * equation_closed
        equation.set_name("tangential_fracture_deformation_equation")
        return equation


class ConstantScaledRadialReturnTangentialContact(ScaledRadialReturnTangentialContact):
    """Scaled radial return tangential contact with user-defined scaling exponent."""

    def scaling_exponent_radial_return(self, subdomains) -> pp.ad.Operator:
        """Scaling exponent for the radial return projection."""
        exponent = pp.ad.Scalar(self.params["contact"]["tangential_scaling_exponent"])
        return exponent


class RandomScaledRadialReturnTangentialContact(ScaledRadialReturnTangentialContact):
    """Scaled radial return tangential contact with random scaling exponent."""

    def before_nonlinear_iteration(self) -> None:
        if not hasattr(self, "random_scaling_exponent"):
            self.random_scaling_exponent = pp.ad.Scalar(1.0)

        self.random_scaling_exponent.set_value(
            np.clip(np.random.normal(0, 1) ** 2, None, 1.0)
        )

    def solver_info(self) -> dict[str, float]:
        """Return solver info for logging."""
        return {
            "random_scaling_exponent": self.random_scaling_exponent.value(
                self.equation_system
            )
        }

    def scaling_exponent_radial_return(self, subdomains) -> pp.ad.Operator:
        """Scaling exponent for the radial return projection."""
        if not hasattr(self, "random_scaling_exponent"):
            self.random_scaling_exponent = pp.ad.Scalar(
                np.clip(np.random.normal(0, 1) ** 2, None, 1.0)
            )

        return self.random_scaling_exponent


class DecayingScaledRadialReturnTangentialContact(ScaledRadialReturnTangentialContact):
    """Scaled radial return tangential contact with scaling exponent decaying to 0
    away from the feasible region.

    """

    def scaling_exponent_radial_return(self, subdomains) -> pp.ad.Operator:
        """Scaling exponent for the radial return projection."""
        # Some functions.
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # Tangential component of the contact traction, and its norm.
        nd_vec_to_tangential = self.tangential_component(subdomains)
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        norm_t_t = f_norm(t_t)

        # Friction bound.
        num_cells = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        b_p = f_max(self.friction_bound(subdomains), zeros_frac)

        # Exponent, being 1.0 inside the feasible regime, and decaying to 0.0 outside.
        # The zero limit is reached at roughly 2 times the friction bound.
        exponent = f_max(
            pp.ad.Scalar(1.0)
            - pp.ad.Scalar(0.5)
            * f_max(norm_t_t - b_p, zeros_frac)
            / f_max(b_p, pp.ad.Scalar(self.numerical.open_state_tolerance)),
            pp.ad.Scalar(0.0),
        )
        # exponent = f_exp(-(f_max(norm_t_t - b_p, zeros_frac)))  # / (b_p**2))
        return exponent
