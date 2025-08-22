"""Model equations for contact mechanics with normal and tangential contact."""

from abc import abstractmethod
import porepy as pp
import numpy as np

import ncp
from functools import partial


class NCP_NormalContact:
    """NCP formulation for normal contact."""

    @abstractmethod
    def normal_ncp_function(self, force, gap) -> pp.ad.Operator:
        """Return the NCP function for normal contact."""
        raise NotImplementedError("Subclasses must implement normal_ncp_function.")

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """NCP formulation for normal contact."""

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Numerical weight
        c_num = self.contact_mechanics_numerical_constant(subdomains)

        # The normal component of the contact force and the displacement jump
        force = pp.ad.Scalar(-1.0) * t_n
        gap = c_num * (u_n - self.fracture_gap(subdomains))

        equation: pp.ad.Operator = self.normal_ncp_function(force, gap)
        equation.set_name("normal_fracture_deformation_equation")
        return equation


class NCP_TangentialContact:
    """NCP formulation for tangential contact."""

    params: dict
    """Model parameters."""

    @abstractmethod
    def tangential_ncp_function(self, yield_criterion, colinearity) -> pp.ad.Operator:
        """Return the NCP function for tangential contact."""
        raise NotImplementedError("Subclasses must implement tangential_ncp_function.")

    def tangential_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Alternative (NCP) implementation for tangential contact."""

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])

        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components.
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains,
            dim=self.nd - 1,  # type: ignore[call-arg]
        )

        # Map from scalar to the tangential.
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Functions.
        f_sign = pp.ad.Function(ncp.sign, "sign_function")
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        f_abs = pp.ad.Function(partial(pp.ad.l2_norm, 1), "abs_function")

        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise. Need a tolerance due to numerical errors.
        f_characteristic = pp.ad.Function(
            partial(
                pp.ad.functions.characteristic_function,
                self.numerical.open_state_tolerance,
            ),
            "characteristic_function_for_zero_normal_traction",
        )

        # The numerical constant for the contact problem has the task to balance orders
        # of magnitude of the different fields. Essentially, displacements and tractions
        # need to be scaled to be of the same order of magnitude.
        # The numerical parameter is a cell-wise scalar which must be extended to a
        # vector quantity to be used in the equation (multiplied from the right).
        c_num = scalar_to_tangential @ self.contact_mechanics_numerical_constant(
            subdomains
        )
        u_t_increment_scaled = c_num * u_t_increment
        u_t_increment_scaled.set_name("u_t_increment_scaled")

        # Orthogonality condition
        scaled_orthogonality = self.orthogonality(subdomains)

        # Coulomb friction bound
        friction_bound = self.friction_bound(subdomains)

        # Yield criterion
        yield_criterion = self.yield_criterion(subdomains)

        # Colinearity condition
        colinearity = self.colinearity_condition(subdomains)

        # Principled choices for open and closed states (required for NCP formulations)
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        characteristic_open = f_characteristic(f_max(friction_bound, zeros_frac))
        characteristic_open.set_name("characteristic_function_open")
        chi_open = scalar_to_tangential @ characteristic_open
        chi_open.set_name("chi_open")

        ones_frac = pp.ad.DenseArray(np.ones(num_cells))
        characteristic_closed = ones_frac - characteristic_open

        # Characteristic functions for the different cases of singularities.
        characteristic_origin: pp.ad.Operator = characteristic_closed * (
            f_characteristic(f_norm(t_t) + f_norm(u_t_increment_scaled))
        )

        # characteristic_origin_t: pp.ad.Operator = characteristic_closed * (
        #    f_characteristic(f_norm(t_t))
        # )
        # characteristic_origin_t.set_name("characteristic_function_origin_traction_t")

        if self.nd == 2:
            modified_yield_criterion = (
                friction_bound - f_sign(u_t_increment_scaled) * t_t
            )
        elif self.nd == 3:
            assert False, "double check this for 3d"
            modified_yield_criterion = friction_bound - f_sign(
                scaled_orthogonality
            ) * f_norm(t_t)
        else:
            raise NotImplementedError(f"Unknown dimension: {self.nd}")
        characteristic_stick_slip_transition: pp.ad.Operator = (
            characteristic_closed
            * f_characteristic(
                f_abs(modified_yield_criterion) + f_norm(u_t_increment_scaled)
            )
        )
        characteristic_stick_slip_transition.set_name(
            "characteristic_function_stick_slip_transition"
        )
        regularization = self.params.get(
            "stick_slip_regularization",
            "origin",
            # "origin_and_stick_slip_transition",
        )
        match regularization:
            case "origin":
                chi_singular = characteristic_origin

            case "origin_and_stick_slip_transition":
                chi_singular = (
                    characteristic_origin + characteristic_stick_slip_transition
                )

            # case "origin_t_and_stick_slip_transition":
            #    chi_singular = (
            #        characteristic_origin_t + characteristic_stick_slip_transition
            #    )

            case _:
                assert False, f"Unknown complementary_approach: {regularization}"
        chi_closed_singular = scalar_to_tangential @ (
            characteristic_closed * chi_singular
        )
        chi_closed_regular = scalar_to_tangential @ (
            characteristic_closed * (pp.ad.Scalar(1.0) - chi_singular)
        )

        # Open state equation
        open_equation: pp.ad.Operator = t_t
        open_equation.set_name("tangential_open_equation")

        # Closed state equation (singular)
        singular_closed_equation: pp.ad.Operator = (
            # TODO clean up
            # u_t_increment_scaled_to_traction
            # c_num * (u_t - u_t.previous_iteration())
            u_t - u_t.previous_iteration()
        )
        singular_closed_equation.set_name("tangential_singular_closed_equation")

        # Closed state equation (regular)
        complementarity_equation: pp.ad.Operator = self.tangential_ncp_function(
            yield_criterion, colinearity
        )
        complementarity_equation.set_name("tangential_complementarity_equation")
        alignment_equation = self.alignment(subdomains)
        alignment_equation.set_name("tangential_alignment_eq")
        e_0 = tangential_basis[0]
        e_1 = tangential_basis[-1]
        regular_closed_equation: pp.ad.Operator = (
            e_0 @ complementarity_equation + e_1 @ alignment_equation
        )
        regular_closed_equation.set_name("tangential_regular_closed_equation")

        # Combine the equations
        equation: pp.ad.Operator = (
            chi_open * open_equation
            + chi_closed_regular * regular_closed_equation
            + chi_closed_singular * singular_closed_equation
        )
        equation.set_name("tangential_fracture_deformation_equation")
        return equation


class NCP_MIN_NormalContact(NCP_NormalContact):
    """NCP formulation for normal contact."""

    def normal_ncp_function(self, force, gap) -> pp.ad.Operator:
        """Return the NCP function for normal contact."""
        return ncp.min(force, gap)


class NCP_MIN_MU_NormalContact(NCP_NormalContact):
    """NCP formulation for normal contact."""

    params: dict
    """Model parameters."""

    def normal_ncp_function(self, force, gap) -> pp.ad.Operator:
        """Return the NCP function for normal contact."""
        mu = self.params["contact"].get("ncp-regularization", 1e-5)
        return ncp.min(force, gap, mu=mu)


class NCP_FB_NormalContact(NCP_NormalContact):
    """NCP formulation for normal contact using the FB function."""

    def normal_ncp_function(self, force, gap) -> pp.ad.Operator:
        """Return the NCP function for normal contact."""

        # return ncp.min_regularized_fb(force, gap, tol=1e-10)
        return ncp.fb(force, gap)


class NCP_FB_MU_NormalContact(NCP_NormalContact):
    """NCP formulation for normal contact using the FB function."""

    params: dict
    """Model parameters."""

    def normal_ncp_function(self, force, gap) -> pp.ad.Operator:
        """Return the NCP function for normal contact."""
        mu = self.params["contact"].get("ncp-regularization", 1e-5)
        return ncp.fb(force, gap, mu=mu)
        # return ncp.min_regularized_fb(force, gap, tol=1e-10, mu=1e-5)


class NCP_MIN_TangentialContact(NCP_TangentialContact):
    """NCP formulation for tangential contact using the MIN function."""

    def tangential_ncp_function(self, yield_criterion, colinearity) -> pp.ad.Operator:
        """Return the NCP function for tangential contact."""
        return ncp.min(yield_criterion, colinearity)


class NCP_FB_TangentialContact(NCP_TangentialContact):
    """NCP formulation for tangential contact using the FB function."""

    def tangential_ncp_function(self, yield_criterion, colinearity) -> pp.ad.Operator:
        """Return the NCP function for tangential contact."""
        # TODO Test!
        # return ncp.min_regularized_fb(yield_criterion, colinearity, tol=1e-10)
        return ncp.fb(yield_criterion, colinearity)


# TODO clean up!

#        if ncp_type == "min-alternative-stick":
#            slip_equation: pp.ad.Operator = ncp.min(
#                yield_criterion, scaled_orthogonality
#            )
#            stick_equation = pp.ad.Scalar(0.5) * (
#                scaled_orthogonality - f_norm(u_t_increment_scaled) * friction_bound
#            )
#        elif ncp_type == "min-sqrt":
#            stick_term = (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#            reg = 1e-3
#            closed_equation: pp.ad.Operator = ncp.min(
#                yield_criterion,
#                f_sign(stick_term)
#                * (f_abs_reg(stick_term) + pp.ad.Scalar(reg**2)) ** 0.5
#                - pp.ad.Scalar(reg),
#            )
#        elif ncp_type == "min-sqrt-star":
#            stick_term = (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#            reg = 1e-3
#            closed_equation: pp.ad.Operator = pp.ad.Scalar(-1) * f_max(
#                pp.ad.Scalar(-1) * yield_criterion,
#                pp.ad.Scalar(-1)
#                * (
#                    pp.ad.Scalar(1e-6) * stick_term + f_sign(stick_term) * stick_term**2
#                    # * (
#                    #    (f_abs_reg(stick_term) + pp.ad.Scalar(reg**2)) ** 0.5
#                    #    - pp.ad.Scalar(reg)
#                    # )
#                ),
#            )
#        elif ncp_type == "min-log":
#            stick_term = (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#            closed_equation: pp.ad.Operator = ncp.min(
#                yield_criterion,
#                f_sign(stick_term) * f_log(f_abs_reg(stick_term) + pp.ad.Scalar(1.0)),
#            )
#        elif ncp_type == "min-log-reg":
#            stick_term = (
#                scaled_orthogonality
#                - f_norm_reg(u_t_increment_scaled_to_one) * friction_bound
#            )
#            stick_equation = f_sign(stick_term) * f_log(
#                f_abs_reg(stick_term) + pp.ad.Scalar(1.0)
#            )
#            closed_equation: pp.ad.Operator = ncp.min(yield_criterion, stick_equation)
#
#        elif ncp_type == "min-linear":
#            # b_p = f_max(self.friction_bound(subdomains), zeros_frac)
#            modified_yield_criterion = (
#                self.friction_bound(subdomains) - f_sign(u_t_increment) * t_t
#            )
#            char_t_0 = f_characteristic(f_norm(t_t))
#            stick_term = char_t_0 * u_t_increment + (
#                pp.ad.Scalar(1.0) - char_t_0
#            ) * scaled_orthogonality / f_norm(t_t)
#            closed_equation = ncp.min(modified_yield_criterion, stick_term)
#
#        elif ncp_type == "fb-linear":
#            # b_p = f_max(self.friction_bound(subdomains), zeros_frac)
#            modified_yield_criterion = (
#                self.friction_bound(subdomains) - f_sign(u_t_increment) * t_t
#            )
#            char_t_0 = f_characteristic(f_norm(t_t))
#            stick_term = char_t_0 * f_norm(u_t_increment) + f_nan_to_num(
#                (pp.ad.Scalar(1.0) - char_t_0) * scaled_orthogonality / f_norm(t_t)
#            )
#            closed_equation = ncp.min_regularized_fb(
#                modified_yield_criterion, stick_term, tol=1e-10
#            )
#        elif ncp_type == "min-star":
#            switch = self.switch(subdomains)
#            slip_equation: pp.ad.Operator = ncp.min(
#                yield_criterion, scaled_orthogonality
#            )
#            min_stick_equation = pp.ad.Scalar(0.5) * (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#            active_set_stick_equation = u_t_increment_scaled_to_traction
#            stick_equation = (
#                switch * min_stick_equation
#                + (pp.ad.Scalar(1.0) - switch) * active_set_stick_equation
#            )
#        elif ncp_type == "fb-alternative-stick":
#            slip_equation = ncp.min_regularized_fb(
#                yield_criterion, scaled_orthogonality, tol=1e-10
#            )
#            stick_equation = pp.ad.Scalar(0.5) * (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#        elif ncp_type == "min/fb":
#            # min-NCP: min(a,b) = -max(-a,-b)
#            min_slip_equation: pp.ad.Operator = ncp.min(
#                yield_criterion, scaled_orthogonality
#            )
#            fb_slip_equation = ncp.min_regularized_fb(
#                yield_criterion, scaled_orthogonality, tol=1e-10
#            )
#            # stick_equation = orthogonality - f_norm(u_t_increment) * friction_bound
#            fb_stick_equation = pp.ad.Scalar(0.5) * (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#
#            min_stick_equation = pp.ad.Scalar(0.5) * (
#                scaled_orthogonality
#                - f_norm(u_t_increment_scaled_to_one) * friction_bound
#            )
#
#            switch = self.switch(subdomains)
#            slip_equation = (
#                switch * min_slip_equation
#                + (pp.ad.Scalar(1.0) - switch) * fb_slip_equation
#            )
#            stick_equation = (
#                switch * min_stick_equation
#                + (pp.ad.Scalar(1.0) - switch) * fb_stick_equation
#            )
