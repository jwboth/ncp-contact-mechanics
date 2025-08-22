from functools import partial

import numpy as np
import porepy as pp


class RadialReturnTangentialContact:
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
        equation: pp.ad.Operator = t_t - min_term * t_t_trial
        equation.set_name("tangential_fracture_deformation_equation")
        return equation
