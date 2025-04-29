"""Datailed export of iteration-dependent approximations including effective
quantities."""

import logging
from functools import partial

import numpy as np
import porepy as pp

logger = logging.getLogger(__name__)


class IterationExporting:
    """Class for exporting iteration-dependent approximations."""

    @property
    def iterate_indices(self):
        """Force storing all previous iterates."""
        return np.array([0, 1])

    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
            length_scale=self.units.m,
        )

    def data_to_export(self):
        """Add data to regular data export:

        * contact states (physical)
        * contact states (connected to augemented Lagrangian idea)

        """
        data = super().data_to_export()

        # Exclude contact_traction from data and scale it.
        data = [d for d in data if d[1] != "contact_traction"]

        # Add data to the fracture
        not_exported = []
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            scaled_contact_traction = self.characteristic_contact_traction(
                [sd]
            ) * self.contact_traction([sd])
            scaled_contact_traction_n = (
                self.normal_component([sd]) @ scaled_contact_traction
            )
            scaled_contact_traction_t = (
                self.tangential_component([sd]) @ scaled_contact_traction
            )
            f_norm = pp.ad.Function(
                partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
            )
            scaled_contact_traction_t_norm = f_norm(scaled_contact_traction_t)
            slip_tendency = scaled_contact_traction_t_norm / scaled_contact_traction_n
            data.append(
                (
                    sd,
                    "contact_traction",
                    self.units.convert_units(1, "Pa^-1")
                    * scaled_contact_traction.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_n",
                    self.units.convert_units(1, "Pa^-1")
                    * scaled_contact_traction_n.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_t",
                    self.units.convert_units(1, "Pa^-1")
                    * scaled_contact_traction_t.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "slip_tendency",
                    np.absolute(slip_tendency.value(self.equation_system)),
                )
            )

        # Deviation from reference state:
        # * displacement
        # * pressure
        # * interface displacement
        if hasattr(self, "has_reference_momentum_state"):
            for i, sd in enumerate(self.mdg.subdomains(dim=self.nd)):
                reference_displacement = pp.ad.TimeDependentDenseArray(
                    "reference_displacement", [sd]
                )
                displacement_deviation = (
                    self.displacement([sd]) - reference_displacement
                )
                data.append(
                    (
                        sd,
                        "deviation_displacement",
                        displacement_deviation.value(self.equation_system),
                    )
                )
                data.append(
                    (
                        sd,
                        "reference_displacement",
                        reference_displacement.value(self.equation_system),
                    )
                )
        if hasattr(self, "has_reference_flow_state"):
            for i, sd in enumerate(self.mdg.subdomains()):
                reference_pressure = pp.ad.TimeDependentDenseArray(
                    "reference_pressure", [sd]
                )
                pressure_deviation = self.pressure([sd]) - reference_pressure
                data.append(
                    (
                        sd,
                        "deviation_pressure",
                        self.units.convert_units(
                            pressure_deviation.value(self.equation_system),
                            "Pa",
                        )
                    )
                )
                data.append(
                    (
                        sd,
                        "reference_pressure",
                        self.units.convert_units(
                            reference_pressure.value(self.equation_system),
                            "Pa",
                        )
                    )
                )

        if hasattr(self, "has_reference_momentum_state"):
            for intf in self.mdg.interfaces(dim=self.nd - 1):
                reference_interface_displacement = pp.ad.TimeDependentDenseArray(
                    "reference_interface_displacement", [intf]
                )
                interface_displacement_deviation = (
                    self.interface_displacement([intf])
                    - reference_interface_displacement
                )
                data.append(
                    (
                        intf,
                        "deviation_interface_displacement",
                        interface_displacement_deviation.value(self.equation_system),
                    )
                )
                data.append(
                    (
                        intf,
                        "reference_interface_displacement",
                        reference_interface_displacement.value(self.equation_system),
                    )
                )
        if len(not_exported) > 0:
            not_exported = list(set(not_exported))
            logger.info(f"Not all data could be exported. Missing: {not_exported}")

        # Add contact states
        states = self.compute_fracture_states(split_output=True, trial=False)
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            data.append((sd, "states", states[i]))

        return data

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Plots:
            - Variables
            - Variable increments
            - Contact states
            - Time increments of u_t

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        data = []
        variables = self.equation_system.variables
        for var in variables:
            # Note that we use iterate_index=0 to get the current solution, whereas
            # the regular exporter uses time_step_index=0.
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.units.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))

            # Append increments if available
            try:
                prev_scaled_values = self.equation_system.get_variable_values(
                    variables=[var], iterate_index=1
                )
                inc_values = self.units.convert_units(
                    scaled_values - prev_scaled_values, units, to_si=True
                )
            except:
                inc_values = self.units.convert_units(
                    scaled_values - scaled_values, units, to_si=True
                )
            data.append((var.domain, var.name + "_inc", inc_values))

        if False:
            for sd in self.mdg.subdomains(dim=self.nd, return_data=False):
                matrix_perm = self.matrix_permeability([sd])
                data.append(
                    (sd, "matrix_perm", matrix_perm.value(self.equation_system))
                )

        # Add residuals for each subproblem - use variables as a proxy for subproblems
        try:
            _, residual = self.linear_system
            # Fetch all var names
            for var in variables:
                var_dofs = self.equation_system.dofs_of([var])
                data.append((var.domain, var.name + "_equation", residual[var_dofs]))
        except:
            ...

        # Exclude contact_traction from data and scale it.
        data = [d for d in data if d[1] != "contact_traction"]

        # Add contact states and time increments of u_t.
        # Include various apertures and normal permeabilities.
        not_exported = []
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            nd_vec_to_normal = self.normal_component([sd])
            nd_vec_to_tangential = self.tangential_component([sd])

            scaled_contact_traction = self.characteristic_contact_traction(
                [sd]
            ) * self.contact_traction([sd])

            data.append(
                (
                    sd,
                    "contact_traction",
                    scaled_contact_traction.value(self.equation_system),
                )
            )

            # Contact mechanics
            t_n: pp.ad.Operator = nd_vec_to_normal @ scaled_contact_traction
            u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump([sd])
            t_t: pp.ad.Operator = nd_vec_to_tangential @ scaled_contact_traction
            u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump([sd])
            u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
            f_norm = pp.ad.Function(
                partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
            )
            t_t_norm: pp.ad.Operator = f_norm(t_t)
            slip_tendency: pp.ad.Operator = t_t_norm / t_n

            data.append(
                (
                    sd,
                    "u_n",
                    u_n.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "u_t",
                    u_t.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "u_t_increment",
                    u_t_increment.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_n",
                    self.units.convert_units(1, "Pa^-1")
                    * t_n.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_t",
                    self.units.convert_units(1, "Pa^-1")
                    * t_t.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "slip_tendency",
                    slip_tendency.value(self.equation_system),
                )
            )

            try:
                c_n = self.contact_mechanics_numerical_constant(subdomains)
                force = pp.ad.Scalar(-1.0) * t_n
                gap = c_n * (u_n - self.fracture_gap(subdomains))
                f_max = pp.ad.Function(pp.ad.maximum, "max_function")
                ncp_equation_normal = pp.ad.Scalar(-1) * f_max(
                    pp.ad.Scalar(-1) * force,
                    pp.ad.Scalar(-1) * gap,
                )
                data.append(
                    (
                        sd,
                        "ncp_equation_normal",
                        ncp_equation_normal.value(self.equation_system),
                    )
                )
                data.append(
                    (sd, "gap", gap.value(self.equation_system)),
                )
            except:
                not_exported.append("ncp_equation_normal")

            try:
                yield_criterion = self.yield_criterion([sd])
                scaled_orthogonality = self.orthogonality([sd])
                ncp_equation_tangential = pp.ad.Scalar(-1) * f_max(
                    pp.ad.Scalar(-1) * yield_criterion,
                    pp.ad.Scalar(-1) * scaled_orthogonality,
                )
                data.append(
                    (sd, "yield_criterion", yield_criterion.value(self.equation_system))
                )
                data.append(
                    (
                        sd,
                        "orthogonality",
                        scaled_orthogonality.value(self.equation_system),
                    )
                )
                data.append(
                    (
                        sd,
                        "ncp_equation_tangential",
                        ncp_equation_tangential.value(self.equation_system),
                    )
                )
            except:
                not_exported.append("ncp_equation_tangential")

            try:
                normal_fracture_deformation_equation = (
                    self.normal_fracture_deformation_equation([sd])
                )
                tangential_fracture_deformation_equation = (
                    self.tangential_fracture_deformation_equation([sd])
                )
                data.append(
                    (
                        sd,
                        "normal_fracture_deformation_equation",
                        normal_fracture_deformation_equation.value(
                            self.equation_system
                        ),
                    )
                )
                data.append(
                    (
                        sd,
                        "tangential_fracture_deformation_equation",
                        tangential_fracture_deformation_equation.value(
                            self.equation_system
                        ),
                    )
                )
            except:
                not_exported.append("fracture_deformation_equation")

            try:
                f_norm = pp.ad.Function(
                    partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
                )
                tangential_basis: list[pp.ad.SparseArray] = self.basis(
                    [sd],
                    dim=self.nd - 1,  # type: ignore[call-arg]
                )
                scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
                c_num_to_one = self.contact_mechanics_numerical_constant_t(subdomains)
                characteristic_origin = f_norm(
                    (scalar_to_tangential @ c_num_to_one) * u_t_increment
                ) + f_norm(t_t)
                data.append(
                    (
                        sd,
                        "characteristic_origin",
                        characteristic_origin.value(self.equation_system),
                    )
                )
            except:
                not_exported.append("characteristic_origin")

            try:
                alignment = self.alignment([sd])
                data.append((sd, "alignment", alignment.value(self.equation_system)))
            except:
                not_exported.append("alignment")

            # Fluid flow
            try:
                aperture = self.aperture([sd])
                fracture_gap = self.fracture_gap([sd])

                # Append data
                data.append((sd, "aperture", aperture.value(self.equation_system)))
                data.append(
                    (sd, "fracture_gap", fracture_gap.value(self.equation_system))
                )
            except:
                not_exported.append("aperture and fracture gap")

            try:
                perm = self.permeability([sd])
                data.append((sd, "perm", perm.value(self.equation_system)))
            except:
                not_exported.append("permeability")

        # Add contact states
        try:
            states = self.compute_fracture_states(split_output=True)
            try:
                prev_states = self.prev_states.copy()
            except:
                prev_states = states.copy()
            for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
                data.append((sd, "states", states[i]))
                data.append((sd, "prev states", prev_states[i]))
            # Cache contact states
            self.prev_state = states.copy()
        except:
            not_exported.append("contact states")

        not_exported = list(set(not_exported))
        if len(not_exported) > 0:
            logger.warning(f"Not all data could be exported. Missing: {not_exported}")

        return data

    def reset_cycling_analysis(self):
        """Clean up all cached data for cycling analysis."""

        if hasattr(self, "cached_contact_states"):
            del self.cached_contact_states
        if hasattr(self, "cached_contact_vars"):
            del self.cached_contact_vars
        if hasattr(self, "stagnating_states"):
            del self.stagnating_states
        if hasattr(self, "cycling_window"):
            del self.cycling_window
        if hasattr(self, "cached_contact_normal_residuals"):
            del self.cached_contact_normal_residuals
        if hasattr(self, "cached_contact_tangential_residuals"):
            del self.cached_contact_tangential_residuals
        # if hasattr(self, "previous_states"):
        #    assert hasattr(self, "states")
        #    self.previous_states = self.states.copy()

    def check_cycling(self):
        """Check for cycling in contact states."""

        # Initialize cache
        if not hasattr(self, "cached_contact_states"):
            self.cached_contact_states = []
        if not hasattr(self, "cached_contact_vars"):
            self.cached_contact_vars = []
        if not hasattr(self, "cached_contact_normal_residuals"):
            self.cached_contact_normal_residuals = []
        if not hasattr(self, "cached_contact_tangential_residuals"):
            self.cached_contact_tangential_residuals = []
        cycling = False
        cycling_window = 0
        stagnating_states = False

        # Fetch states and variables
        self.states = self.compute_fracture_states(split_output=True)
        vars = self.fetch_fracture_vars()
        normal_residuals, tangential_residuals = self.fetch_fracture_residuals()

        # Determine change in time, i.e., the total difference between states and
        # previous_states
        try:
            total_changes_in_time = np.count_nonzero(
                np.logical_not(
                    np.isclose(
                        np.concatenate(self.states),
                        np.concatenate(self.previous_states),
                    )
                )
            )
        except:
            total_changes_in_time = 0
        logger.info(f"Changes in time: {total_changes_in_time}")

        self.nonlinear_solver_statistics.total_contact_state_changes_in_time = (
            total_changes_in_time
        )

        # Check for stagnation in contact states
        required_length = 12
        if len(self.cached_contact_states) >= required_length:
            stagnating_states = True
            for i in range(required_length):
                if not np.allclose(
                    np.concatenate(self.states),
                    np.concatenate(self.cached_contact_states[-i - 1]),
                ):
                    stagnating_states = False
                    break
        if stagnating_states:
            logger.info(f"Stagnating states detected.")

        elif len(self.cached_contact_states) > 0:
            # Determine detailed contact state changes
            changes = np.zeros((3, 3), dtype=int)
            num_contact_states = np.zeros(3, dtype=int)
            try:
                for i in range(3):
                    num_contact_states[i] = int(
                        np.sum(np.concatenate(self.states) == i)
                    )
                    for j in range(3):
                        changes[i, j] = int(
                            np.sum(
                                np.logical_and(
                                    np.concatenate(self.states) == i,
                                    np.concatenate(self.cached_contact_states[-1]) == j,
                                )
                            )
                        )
            except:
                ...
            logger.info(f"Changes in states: \n{changes}")

            # Count general changes
            try:
                total_changes = np.count_nonzero(
                    np.logical_not(
                        np.isclose(
                            np.concatenate(self.states),
                            np.concatenate(self.cached_contact_states[-1]),
                        )
                    )
                )
            except:
                total_changes = 0
            logger.info(f"Total changes: {total_changes}")

            # Check if changes are small
            if total_changes < 7:
                self.small_changes = True
            else:
                self.small_changes = False

            # Monitor contact state changes
            self.nonlinear_solver_statistics.num_contact_states = (
                num_contact_states.tolist()
            )
            self.nonlinear_solver_statistics.contact_state_changes = changes.tolist()
            self.nonlinear_solver_statistics.total_contact_state_changes = total_changes
            if total_changes > 0:
                self.nonlinear_solver_statistics.last_update_contact_states = (
                    self.nonlinear_solver_statistics.num_iteration
                )
            if hasattr(self, "update_num_contact_states_changes"):
                self.update_num_contact_states_changes()

        # Check for cycling based on closedness of states and variables
        rtol = 1e-2
        for i in range(len(self.cached_contact_states) - 1, 1, -1):
            if self.states != [] and (
                np.allclose(
                    np.concatenate(self.states),
                    np.concatenate(self.cached_contact_states[i]),
                )
                and np.allclose(
                    np.concatenate(vars),
                    np.concatenate(self.cached_contact_vars[i]),
                    rtol=rtol,
                )
                and np.allclose(
                    np.concatenate(self.cached_contact_states[-1]),
                    np.concatenate(self.cached_contact_states[i - 1]),
                )
                and np.allclose(
                    np.concatenate(self.cached_contact_vars[-1]),
                    np.concatenate(self.cached_contact_vars[i - 1]),
                    rtol=rtol,
                )
            ):
                cycling = True
                cycling_window = len(self.cached_contact_states) - i

                logger.info(f"Cycling detected with window {cycling_window}.")

            if cycling:
                break
        self.cached_contact_states.append(self.states)
        self.cached_contact_vars.append(vars)
        self.cached_contact_normal_residuals.append(normal_residuals)
        self.cached_contact_tangential_residuals.append(tangential_residuals)

        # Clean up cache
        if len(self.cached_contact_states) > 10:
            self.cached_contact_states.pop(0)
            self.cached_contact_vars.pop(0)
            self.cached_contact_normal_residuals.pop(0)
            self.cached_contact_tangential_residuals.pop(0)

        # Store cycling window
        if cycling_window > 0:
            self.cycling_window = cycling_window
        else:
            self.cycling_window = 0

        # Store stagnating status
        self.stagnating_states = stagnating_states

        # Monitor as part of nonlinear solver statistics
        self.nonlinear_solver_statistics.cycling_window = self.cycling_window
        self.nonlinear_solver_statistics.stagnating_states = self.stagnating_states

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing
        between iterations and time steps in the regular exporter's
        history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with
        # the time part, we multiply the latter by the next power of ten above
        # the maximum number of nonlinear iterations. Default value set to 10
        # in accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        r = 10
        while r <= n:
            r *= 10
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iteration
            + r * self.time_manager.time_index,
        )

    def before_nonlinear_loop(self):
        self.previous_states = self.compute_fracture_states(split_output=True)
        super().before_nonlinear_loop()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution
        to iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()
        self.check_cycling()
        # print()  # force progressbar to output.

    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()
        self.reset_cycling_analysis()
