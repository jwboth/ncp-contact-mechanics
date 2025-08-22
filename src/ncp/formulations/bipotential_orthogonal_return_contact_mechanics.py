import porepy as pp
from functools import partial
import numpy as np


class BipotentialOrthogonalReturnContact:
    """The bipotential orthogonal return contact mechanics formulation.

    It is based on the de Saxce-Feng formulation/bipotential theory, which
    uses a different augmentation of the contact traction than the radial
    return formulation. The project follows then a orthogonal return map
    onto the Coulomb cone.

    The bipotential formulation does not separate between normal
    and tangential equations.

    """

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        # Quantities required for the projection.
        scalar, chi, t = self._auxiliary_orthogonal_contact_projection(subdomains)

        # Characteristic functions
        characteristic = chi["chi_outside_dual_cone"]

        # Projection of the augmented traction (normal component).
        projection_n = characteristic * (
            t["t_n_augmented"] - scalar["weight"] * scalar["mu"]
        )

        # Assignment of traction to the projection of the augmented traction (normal component)
        equation = t["t_n"] - projection_n
        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints."""

        # Quantities required for the projection.
        scalar, chi, t = self._auxiliary_orthogonal_contact_projection(subdomains)

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

        equation = t["t_t"] - projection_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def _auxiliary_orthogonal_contact_projection(self, subdomains: list[pp.Grid]):
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
