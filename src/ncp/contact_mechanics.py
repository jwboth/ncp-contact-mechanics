"""Model equations for contact mechanics with normal and tangential contact."""

from functools import partial

import numpy as np
import porepy as pp

import ncp


class ScaledContact:
    def cnum(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Numerical constant for the contact problem [-]."""
        return pp.ad.Scalar(
            self.params.get("contact_mechanics_numerical_constant_scaling", 1.0)
        )

    def characteristic_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Characteristic distance for the contact problem [m].

        Needed for NCP formulations.

        """
        return pp.ad.Scalar(self.solid.residual_aperture + self.solid.fracture_gap)

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """Scale the original numerical constant [m^-1]."""
        return self.cnum(subdomains) * super().contact_mechanics_numerical_constant(
            subdomains
        )

    def contact_mechanics_numerical_constant_t(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """Numerical constant for the contact problem [m^-1].

        As the normal contact, but without the shear modulus.

        """
        # characteristic_distance = self.characteristic_jump(subdomains)
        characteristic_distance = self.characteristic_displacement(subdomains)
        # val = self.cnum_t(subdomains) / characteristic_distance
        val = self.cnum(subdomains) / characteristic_distance
        return val


class UnscaledContact(ScaledContact):
    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Scalar:
        """Numerical constant for the contact problem [Pa * m^-1].

        A physical interpretation of this constant is as an elastic modulus for the
        fracture, as it appears as a scaling of displacement jumps when comparing to
        contact tractions.

        Parameters:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant, as scalar.

        """
        cnum = self.cnum(subdomains)
        youngs_modulus = self.youngs_modulus(subdomains)
        size = pp.ad.Scalar(np.max(self.domain.side_lengths()))
        val = cnum * youngs_modulus / size
        val.set_name("Contact_mechanics_numerical_constant")
        return val

    def characteristic_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Scaling factor for the contact tractions [Pa]."""
        return pp.ad.Scalar(1.0)

    def characteristic_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Scaling factor for the jump operator [m]."""
        return pp.ad.Scalar(1.0)


class NCPNormalContact:
    """NCP implementation for normal contact."""

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Alternative (NCP) implementation for normal contact."""

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Numerical weight
        c_num_to_traction = self.contact_mechanics_numerical_constant(subdomains)

        # The normal component of the contact force and the displacement jump
        force = pp.ad.Scalar(-1.0) * t_n
        gap = c_num_to_traction * (u_n - self.fracture_gap(subdomains))

        ncp_type = self.params.get("ncp_type", "min")
        assert ncp_type in [
            # Classical NCP formulations
            "min",
            "fb-partial",
            "fb",
            # FB without regularization
            "fb-partial-star",
            "fb-star",
            # Linearized NCP formulations
            "min-linear",
            "fb-linear",
            # Alternative min-NCP formulations
            "min-alternative-stick",
            "min-log",
            "min-sqrt",
            # Turn off intersections
            "min-no-intersections",
        ]

        if ncp_type in [
            "min",
            "fb-partial",
            "fb-partial-star",
            "min-linear",
            "fb-linear",
            "min-alternative-stick",
            "min-log",
            "min-log-reg",
            "min-sqrt",
            "min-no-intersections",
        ]:
            # min-NCP: min(a,b) = -max(-a,-b)
            equation: pp.ad.Operator = ncp.min(force, gap)
        elif ncp_type in [
            "fb",
        ]:
            # Fischer-Burmeister: (a**2 + b**2)**0.5 - (a + b)
            equation = ncp.min_regularized_fb(force, gap, tol=1e-10)
        elif ncp_type in [
            "fb-star",
        ]:
            # Fischer-Burmeister: (a**2 + b**2)**0.5 - (a + b)
            equation = ncp.fb(force, gap)
        else:
            raise NotImplementedError(f"Unknown ncp_type: {ncp_type}")

        equation.set_name("normal_fracture_deformation_equation")
        return equation


class AuxiliaryContact:
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

    def orthogonality(self, subdomains: list[pp.Grid], scaled=True):
        """t_t * u_t_increment."""

        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Auxiliary operator for scalar products
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains,
            dim=self.nd - 1,  # type: ignore[call-arg]
        )
        nd_to_scalar_sum = pp.ad.sum_operator_list(
            [e.T for e in tangential_basis]  # type: ignore[call-arg]
        )

        c_num_to_one_as_scalar = self.contact_mechanics_numerical_constant_t(subdomains)
        c_num_to_one = pp.ad.sum_operator_list(
            [e_i * c_num_to_one_as_scalar * e_i.T for e_i in tangential_basis]
        )

        # Orthogonality condition
        if scaled:
            orthogonality = nd_to_scalar_sum @ (t_t * (c_num_to_one @ u_t_increment))
            orthogonality.set_name("orthogonality")
        else:
            orthogonality = nd_to_scalar_sum @ (t_t * u_t_increment)
            orthogonality.set_name("unscaled orthogonality")
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
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        # Equivalent to using the time derivative of u_t, use the time increment
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Compute the determinant of the two vectors
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains,
            dim=self.nd - 1,  # type: ignore[call-arg]
        )

        c_num_to_one_as_scalar = self.contact_mechanics_numerical_constant_t(subdomains)
        c_num_to_one = pp.ad.sum_operator_list(
            [e_i * c_num_to_one_as_scalar * e_i.T for e_i in tangential_basis]
        )
        scaled_u_t_increment = c_num_to_one @ u_t_increment

        e_0 = tangential_basis[0]
        e_1 = tangential_basis[1]
        det: pp.ad.Operator = (e_0.T @ scaled_u_t_increment) * (e_1.T @ t_t) - (
            e_1.T @ scaled_u_t_increment
        ) * (e_0.T @ t_t)
        det.set_name("determinant")
        # f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        # det = self.orthogonality(subdomains) - f_norm(t_t) * f_norm(
        #    scaled_u_t_increment
        # )
        return det


class NCPTangentialContact2d:
    def tangential_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Alternative (NCP) implementation for tangential contact."""

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        # Ignore mypy complaint on unknown keyword argument
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains,
            dim=self.nd - 1,  # type: ignore[call-arg]
        )

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell. EK: mypy insists
        # that the argument to sum should be a list of booleans. Ignore this error.
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = scalar_to_tangential @ pp.ad.DenseArray(np.ones(num_cells))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        # Functions EK: Should we try to agree on a name convention for ad functions?
        # EK: Yes. Suggestions?
        f_sign = pp.ad.Function(ncp.sign, "sign_function")
        f_nan_to_num = pp.ad.Function(ncp.nan_to_num, "nan_to_num")
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        f_norm_reg = pp.ad.Function(
            partial(ncp.l2_norm_reg, self.nd - 1), "norm_function"
        )
        f_abs = pp.ad.Function(partial(pp.ad.l2_norm, 1), "abs_function")
        f_abs_reg = pp.ad.Function(ncp.abs_reg, "abs_function")
        f_log_reg = pp.ad.Function(ncp.log_reg, "regularzied_log")
        f_log = pp.ad.Function(pp.ad.log, "log")

        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases.
        tol = self.numerical.open_state_tolerance
        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise.
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        # The numerical parameter is a cell-wise scalar which must be extended to a
        # vector quantity to be used in the equation (multiplied from the right).
        # Spelled out, from the right: Restrict the vector quantity to one dimension in
        # the tangential plane (e_i.T), multiply with the numerical parameter, prolong
        # to the full vector quantity (e_i), and sum over all all directions in the
        # tangential plane. EK: mypy insists that the argument to sum should be a list
        # of booleans. Ignore this error.

        # The numerical constant for the contact problem has the task to balance orders
        # of magnitude of the different fields. Essentially, displacements and tractions
        # need to be scaled to be of the same order of magnitude. For different uses,
        # the numerical constant may be scaled differently.
        # c_num_to_traction essentially scales displacement increments to be of unit of tractions
        c_num_to_traction_as_scalar = self.contact_mechanics_numerical_constant(
            subdomains
        )
        c_num_to_traction = pp.ad.sum_operator_list(
            [e_i * c_num_to_traction_as_scalar * e_i.T for e_i in tangential_basis]
        )
        # c_num_to_one_as_scalar essentially scales the tangential displacement increment to be of unit 1
        c_num_to_one_as_scalar = self.contact_mechanics_numerical_constant_t(subdomains)
        c_num_to_one = pp.ad.sum_operator_list(
            [e_i * c_num_to_one_as_scalar * e_i.T for e_i in tangential_basis]
        )

        # Coulomb friction bound
        friction_bound = self.friction_bound(subdomains)

        # Yield criterion
        yield_criterion = self.yield_criterion(subdomains)
        if self.nd == 2:
            modified_yield_criterion = friction_bound - f_sign(u_t_increment) * t_t
        elif self.nd == 3:
            # Auxiliary operator for scalar products
            tangential_basis: list[pp.ad.SparseArray] = self.basis(
                subdomains,
                dim=self.nd - 1,  # type: ignore[call-arg]
            )
            nd_to_scalar_sum = pp.ad.sum_operator_list(
                [e.T for e in tangential_basis]  # type: ignore[call-arg]
            )
            modified_yield_criterion = friction_bound - f_sign(
                nd_to_scalar_sum @ (u_t_increment * t_t)
            ) * f_norm(t_t)
        else:
            raise NotImplementedError(f"Unknown dimension: {self.nd}")

        # Orthogonality condition
        scaled_orthogonality = self.orthogonality(subdomains, True)
        orthogonality = self.orthogonality(subdomains, False)

        # Principled choices for open, stick, slip
        characteristic_open = self.contact_mechanics_open_state_characteristic(
            subdomains
        )
        characteristic_closed = (
            ones_frac - self.contact_mechanics_open_state_characteristic(subdomains)
        )
        characteristic_slip: pp.ad.Operator = (ones_frac - characteristic_open) * (
            scalar_to_tangential @ f_characteristic(f_max(yield_criterion, zeros_frac))
        )
        characteristic_stick: pp.ad.Operator = (ones_frac - characteristic_open) * (
            ones_frac - characteristic_slip
        )

        characteristic_open.set_name("characteristic_function_open")
        characteristic_closed.set_name("characteristic_function_closed")
        characteristic_slip.set_name("characteristic_function_slip")
        characteristic_stick.set_name("characteristic_function_stick")

        characteristic_origin: pp.ad.Operator = (ones_frac - characteristic_open) * (
            scalar_to_tangential
            @ (
                f_characteristic(
                    f_norm(t_t) + f_norm(c_num_to_traction @ u_t_increment)
                )
            )
        )
        characteristic_origin.set_name("characteristic_function_origin")

        characteristic_stick_slip_transition: pp.ad.Operator = (
            ones_frac - characteristic_open
        ) * (
            scalar_to_tangential
            @ (
                f_characteristic(
                    f_abs(modified_yield_criterion)
                    + f_norm(c_num_to_traction @ u_t_increment)
                )
            )
        )
        characteristic_stick_slip_transition.set_name(
            "characteristic_function_stick_slip_transition"
        )

        # Principled choice for the tangential contact: admissible tangent traction and aligned
        # tangential displacement and traction
        ncp_type = self.params.get("ncp_type", "min")

        if ncp_type == "min-alternative-stick":
            slip_equation: pp.ad.Operator = ncp.min(
                yield_criterion, scaled_orthogonality
            )
            stick_equation = pp.ad.Scalar(0.5) * (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
        elif ncp_type == "min":
            closed_equation: pp.ad.Operator = ncp.min(
                yield_criterion,
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound,
            )

        elif ncp_type == "min-sqrt":
            stick_term = (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
            reg = 1e-3
            closed_equation: pp.ad.Operator = ncp.min(
                yield_criterion,
                f_sign(stick_term)
                * (f_abs_reg(stick_term) + pp.ad.Scalar(reg**2)) ** 0.5
                - pp.ad.Scalar(reg),
            )
        elif ncp_type == "min-sqrt-star":
            stick_term = (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
            reg = 1e-3
            closed_equation: pp.ad.Operator = pp.ad.Scalar(-1) * f_max(
                pp.ad.Scalar(-1) * yield_criterion,
                pp.ad.Scalar(-1)
                * (
                    pp.ad.Scalar(1e-6) * stick_term + f_sign(stick_term) * stick_term**2
                    # * (
                    #    (f_abs_reg(stick_term) + pp.ad.Scalar(reg**2)) ** 0.5
                    #    - pp.ad.Scalar(reg)
                    # )
                ),
            )
        elif ncp_type == "min-log":
            stick_term = (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
            closed_equation: pp.ad.Operator = ncp.min(
                yield_criterion,
                f_sign(stick_term) * f_log(f_abs_reg(stick_term) + pp.ad.Scalar(1.0)),
            )
        elif ncp_type == "min-log-reg":
            stick_term = (
                scaled_orthogonality
                - f_norm_reg(c_num_to_one @ u_t_increment) * friction_bound
            )
            stick_equation = f_sign(stick_term) * f_log(
                f_abs_reg(stick_term) + pp.ad.Scalar(1.0)
            )
            closed_equation: pp.ad.Operator = ncp.min(yield_criterion, stick_equation)

        elif ncp_type == "min-linear":
            # b_p = f_max(self.friction_bound(subdomains), zeros_frac)
            modified_yield_criterion = (
                self.friction_bound(subdomains) - f_sign(u_t_increment) * t_t
            )
            char_t_0 = f_characteristic(f_norm(t_t))
            stick_term = char_t_0 * u_t_increment + (
                pp.ad.Scalar(1.0) - char_t_0
            ) * scaled_orthogonality / f_norm(t_t)
            closed_equation = ncp.min(modified_yield_criterion, stick_term)

        elif ncp_type in ["fb-partial", "fb"]:
            # Fischer-Burmeister: (a**2 + b**2)**0.5 - (a + b)
            stick_term = (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
            closed_equation = ncp.min_regularized_fb(
                yield_criterion, stick_term, tol=1e-10
            )
        elif ncp_type in ["fb-partial-star", "fb-star"]:
            # Fischer-Burmeister: (a**2 + b**2)**0.5 - (a + b)
            stick_term = (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
            closed_equation = ncp.fb(
                yield_criterion,
                stick_term,
            )
        elif ncp_type == "fb-linear":
            # b_p = f_max(self.friction_bound(subdomains), zeros_frac)
            modified_yield_criterion = (
                self.friction_bound(subdomains) - f_sign(u_t_increment) * t_t
            )
            char_t_0 = f_characteristic(f_norm(t_t))
            stick_term = char_t_0 * f_norm(u_t_increment) + f_nan_to_num(
                (pp.ad.Scalar(1.0) - char_t_0) * scaled_orthogonality / f_norm(t_t)
            )
            closed_equation = ncp.min_regularized_fb(
                modified_yield_criterion, stick_term, tol=1e-10
            )
        elif ncp_type == "min-active-set":
            # min-NCP: min(a,b) = -max(-a,-b)
            stick_equation = c_num_to_traction @ u_t_increment
            slip_equation: pp.ad.Operator = ncp.min(
                yield_criterion, scaled_orthogonality
            )
        elif ncp_type == "min-star":
            switch = self.switch(subdomains)
            slip_equation: pp.ad.Operator = ncp.min(
                yield_criterion, scaled_orthogonality
            )
            min_stick_equation = pp.ad.Scalar(0.5) * (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
            active_set_stick_equation = c_num_to_traction @ u_t_increment
            stick_equation = (
                switch * min_stick_equation
                + (pp.ad.Scalar(1.0) - switch) * active_set_stick_equation
            )
        elif ncp_type == "fb-alternative-stick":
            slip_equation = ncp.min_regularized_fb(
                yield_criterion, scaled_orthogonality, tol=1e-10
            )
            stick_equation = pp.ad.Scalar(0.5) * (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )
        elif ncp_type == "min/fb":
            # min-NCP: min(a,b) = -max(-a,-b)
            min_slip_equation: pp.ad.Operator = ncp.min(
                yield_criterion, scaled_orthogonality
            )
            fb_slip_equation = ncp.min_regularized_fb(
                yield_criterion, scaled_orthogonality, tol=1e-10
            )
            # stick_equation = orthogonality - f_norm(u_t_increment) * friction_bound
            fb_stick_equation = pp.ad.Scalar(0.5) * (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )

            min_stick_equation = pp.ad.Scalar(0.5) * (
                scaled_orthogonality
                - f_norm(c_num_to_one @ u_t_increment) * friction_bound
            )

            switch = self.switch(subdomains)
            slip_equation = (
                switch * min_slip_equation
                + (pp.ad.Scalar(1.0) - switch) * fb_slip_equation
            )
            stick_equation = (
                switch * min_stick_equation
                + (pp.ad.Scalar(1.0) - switch) * fb_stick_equation
            )

        else:
            raise ValueError(f"Unknown ncp_type: {ncp_type}")
        # TODO clean up! do not use try-except
        try:
            slip_equation.set_name("tangential_slip_equations")
        except:
            ...

        # Copy closed equation to slip and stick if not defined
        if "stick_equation" not in locals():
            stick_equation = closed_equation
        if "slip_equation" not in locals():
            slip_equation = closed_equation

        regularization = self.params.get("stick_slip_regularization")
        match regularization:
            case "none":
                _characteristic_singular = pp.ad.Scalar(0.0)

            case "origin":
                _characteristic_singular = characteristic_origin

            case "stick_slip_transition":
                _characteristic_singular = characteristic_stick_slip_transition

            case "origin_and_stick_slip_transition":
                _characteristic_singular = (
                    characteristic_origin + characteristic_stick_slip_transition
                )

            case _:
                assert False, f"Unknown complementary_approach: {regularization}"

        e_0 = tangential_basis[0]
        e_1 = tangential_basis[-1]
        equation: pp.ad.Operator = (
            characteristic_open * t_t
            + characteristic_stick * (e_0 @ stick_equation)
            + characteristic_slip * (e_0 @ slip_equation)
            + characteristic_closed * (e_1 @ self.alignment(subdomains))
            + _characteristic_singular * (u_t - u_t.previous_iteration())
            # TODO use regularization parameter here? e.g. c_num_to_traction * 1e-10?
            # * (c_num_to_traction @ (u_t - u_t.previous_iteration()))
        )

        equation.set_name("tangential_fracture_deformation_equation")
        return equation


class LinearRadialReturnTangentialContact:
    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints."""

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell. EK: mypy insists
        # that the argument to sum should be a list of booleans. Ignore this error.
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Variables: The tangential component of the contact traction and the plastic
        # displacement jump.
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # The numerical constant is used to loosen the sensitivity in the transition
        # between sticking and sliding.
        # Expanding using only left multiplication to with scalar_to_tangential does not
        # work for an array, unlike the operators below. Arrays need right
        # multiplication as well.
        c_num = self.contact_mechanics_numerical_constant(subdomains)

        # Combine the above into expressions that enter the equation. c_num will
        # effectively be a sum of SparseArrays, thus we use a matrix-vector product @
        tangential_sum = t_t + (scalar_to_tangential @ c_num) * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        # For the use of @, see previous comment.
        min_term = scalar_to_tangential @ (
            pp.ad.Scalar(-1.0)
            * f_max(
                pp.ad.Scalar(-1.0) * ones_frac,
                pp.ad.Scalar(-1.0) * b_p / norm_tangential_sum,
            )
        )

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = t_t - min_term * tangential_sum
        equation.set_name("tangential_fracture_deformation_equation")
        return equation
