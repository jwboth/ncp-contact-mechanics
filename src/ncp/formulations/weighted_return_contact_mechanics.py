import porepy as pp
from functools import partial
import numpy as np


class WeightedReturnContact:
    @property
    def default_weight_return_map(self) -> float:
        """Default value for the weight in the weighted return map."""
        return 0.5

    def set_weight_return_map(self) -> None:
        self.weight = pp.ad.Scalar(self.default_weight_return_map, "weight")

    def randomize_weight_return_map(self) -> None:
        if not hasattr(self, "weight"):
            self.set_weight_return_map()
        self.weight.set_value(np.random.normal(self.default_weight_return_map, 0.5))

    def normal_bipotential_projection(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        # Quantities required for the projection.
        scalar, chi, t = self._bipotential_utils(subdomains)

        # Characteristic functions
        characteristic = chi["chi_outside_dual_cone"]

        # Projection of the augmented traction (normal component).
        projection_n = characteristic * (
            t["t_n_augmented"] - scalar["weight"] * scalar["mu"]
        )

        return projection_n

    def normal_radial_return_projection(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)

        # The normal component of the contact traction and the displacement jump.
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")

        # The complimentarity condition
        projection = -max_function(
            pp.ad.Scalar(-1.0) * t_n
            - self.contact_mechanics_numerical_constant(subdomains)
            * (u_n - self.fracture_gap(subdomains)),
            zeros_frac,
        )
        return projection

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        # The normal component of the contact traction and the displacement jump.
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)

        # Setup of weight
        if not hasattr(self, "weight"):
            self.set_weight_return_map()

        # Assignment of traction to the projection of the augmented traction (normal component)
        bipotential_projection = self.normal_bipotential_projection(subdomains)
        radial_return_projection = self.normal_radial_return_projection(subdomains)
        equation = (
            t_n
            - self.weight * bipotential_projection
            - (pp.ad.Scalar(1.0) - self.weight) * radial_return_projection
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_bipotential_projection(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        # Quantities required for the projection.
        scalar, chi, t = self._bipotential_utils(subdomains)

        # Projection of the augmented traction (tangential component).
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")

        # Characteristic functions
        characteristic = scalar_to_tangential @ chi["chi_outside_dual_cone"]
        chi_non_origin = pp.ad.Scalar(1.0) - chi["chi_origin"]

        # Projection
        cut_off = pp.ad.Scalar(self.numerical.open_state_tolerance)
        projection_t = characteristic * (
            t["t_t_augmented"]
            - scalar_to_tangential
            @ (
                chi_non_origin
                * scalar["weight"]
                / f_max(f_norm(t["t_t_augmented"]), cut_off)
            )
            * t["t_t_augmented"]
        )
        return projection_t

    def tangential_radial_return_projection(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
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

        # Cut off negative values to avoid open state.
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        b_p = f_max(self.friction_bound(subdomains), zeros_frac)

        # Define the traction to be the linear radial return projection of the
        # augmented traction.
        ones_frac = pp.ad.DenseArray(np.ones(num_cells))
        min_term = scalar_to_tangential @ (
            -characteristic
            * f_max(
                pp.ad.Scalar(-1.0) * ones_frac,
                pp.ad.Scalar(-1.0) * b_p / norm_t_t_trial,
            )
        )
        projection: pp.ad.Operator = min_term * t_t_trial
        return projection

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints."""
        # Variables
        nd_vec_to_tangential = self.tangential_component(subdomains)
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)

        # Setup of weight
        if not hasattr(self, "weight"):
            self.set_weight_return_map()

        # Projections
        bipotential_projection = self.tangential_bipotential_projection(subdomains)
        radial_return_projection = self.tangential_radial_return_projection(subdomains)
        equation = (
            t_t
            - self.weight * bipotential_projection
            - (pp.ad.Scalar(1.0) - self.weight) * radial_return_projection
        )
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def _bipotential_utils(self, subdomains: list[pp.Grid]):
        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        nd_vec_to_tangential = self.tangential_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Functions.
        f_abs = pp.ad.Function(partial(pp.ad.l2_norm, 1), "abs_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_characteristic = pp.ad.Function(
            partial(
                pp.ad.functions.characteristic_function,
                self.numerical.open_state_tolerance,
            ),
            "characteristic_function_for_zero_normal_traction",
        )

        # Projections.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Numerical weight
        c_num = self.contact_mechanics_numerical_constant(subdomains)

        # Augmentation.
        mu = self.friction_coefficient(subdomains)
        t_n_augmented = t_n + c_num * (
            u_n - self.fracture_gap(subdomains) + mu * f_norm(u_t_increment)
        )
        t_n_augmented.set_name("t_n_augmented")
        t_t_augmented = t_t + (scalar_to_tangential @ c_num) * u_t_increment
        t_t_augmented.set_name("t_t_augmented")

        # Collection of traction variables.
        t = {
            "t_n": t_n,
            "t_t": t_t,
            "t_n_augmented": t_n_augmented,
            "t_t_augmented": t_t_augmented,
        }

        # Zero array for later comparison.
        num_cells = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        # Characterictic functions. The augmented traction lies outside the dual cone if the
        # tangential part (scaled by the friction coefficient) has larger modulus than the normal part.
        chi_origin = f_characteristic(f_abs(t_n_augmented) + f_norm(t_t_augmented))
        dual_cone_condition = t_n_augmented - mu * f_norm(t_t_augmented)
        chi_outside_dual_cone = (
            f_characteristic(f_max(dual_cone_condition, zeros_frac)) - chi_origin
        )
        chi_outside_dual_cone.set_name("chi_outside_dual_cone")

        # Collection of characteristic functions.
        chi = {
            "chi_outside_dual_cone": chi_outside_dual_cone,
            "chi_origin": chi_origin,
        }

        # Weighting
        weight = f_max(f_norm(t_t_augmented) + mu * t_n_augmented, zeros_frac) / (
            pp.ad.Scalar(1.0) + mu**2
        )

        # Collection of scalar variables.
        scalar = {
            "weight": weight,
            "mu": mu,
        }

        return scalar, chi, t
