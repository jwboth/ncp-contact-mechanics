from functools import partial

import numpy as np
import porepy as pp


class AuxiliaryContact:
    def colinearity_condition(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Scaled condition for colinearity of the tangential displacement and traction. [-]"""

        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Variables: The tangential component of the contact traction and the
        # displacement jump, and the time increment of the displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Scaling
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1
        )
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        c_num = scalar_to_tangential @ self.contact_mechanics_numerical_constant(
            subdomains
        )
        u_t_increment_scaled = c_num * u_t_increment

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        scaled_orthogonality = self.orthogonality(subdomains)
        friction_bound = self.friction_bound(subdomains)
        return scaled_orthogonality - f_norm(u_t_increment_scaled) * friction_bound

    def yield_criterion(self, subdomains: list[pp.Grid]):
        """F|t_n| - |t_t|."""

        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)

        # Yield criterion
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        yield_criterion = self.friction_bound(subdomains) - f_norm(t_t)
        yield_criterion.set_name("yield_criterion")
        return yield_criterion

    def orthogonality(self, subdomains: list[pp.Grid]):
        """t_t * u_t_increment."""

        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Variables: The tangential component of the contact traction and the
        # displacement jump, and the time increment of the displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Tangential basis for projection and scalar product
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1
        )

        # Components of scalar product
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        c_num = scalar_to_tangential @ self.contact_mechanics_numerical_constant(
            subdomains
        )
        u_t_increment_scaled = c_num * u_t_increment
        orthogonality_vector = t_t * u_t_increment_scaled

        # Scalar product
        if self.nd == 2:
            orthogonality = orthogonality_vector
        elif self.nd == 3:
            e_0 = tangential_basis[0]
            e_1 = tangential_basis[-1]
            orthogonality = e_0.T @ orthogonality_vector + e_1.T @ orthogonality_vector
        orthogonality.set_name("orthogonality")
        return orthogonality

    def alignment(self, subdomains: list[pp.Grid]):
        """det(t_t, u_t_increment)."""
        if self.nd == 2:
            num_cells = sum([sd.num_cells for sd in subdomains])
            return pp.ad.DenseArray(np.zeros(num_cells))
        assert self.nd == 3, "Only implemented for 3d"

        # The tangential component of the contact traction and the displacement jump
        nd_vec_to_tangential = self.tangential_component(subdomains)
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        t_t = t_t.previous_iteration()
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # Compute the determinant of the two vectors
        c_num_to_one = self.contact_mechanics_numerical_constant_t(subdomains)
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1
        )
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        scaled_u_t_increment = (scalar_to_tangential @ c_num_to_one) * u_t_increment

        e_0 = tangential_basis[0]
        e_1 = tangential_basis[1]
        det: pp.ad.Operator = (e_0.T @ scaled_u_t_increment) * (e_1.T @ t_t) - (
            e_1.T @ scaled_u_t_increment
        ) * (e_0.T @ t_t)
        det.set_name("determinant")
        return det
