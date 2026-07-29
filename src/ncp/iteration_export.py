"""Datailed export of iteration-dependent approximations including effective
quantities."""

import logging
from functools import partial

import numpy as np
import porepy as pp
from porepy.numerics.solvers.convergence_check import ConvergenceStatus

logger = logging.getLogger(__name__)


class CustomExporting:
    def data_to_export(self):
        """Add data to regular data export:
        * fracture aperture
        * fracture gap
        * scaled contact traction
        * contact states (physical)
        * contact states (connected to augemented Lagrangian idea)

        """
        data = super().data_to_export()

        # Exclude contact_traction from data and scale it.
        data = [d for d in data if d[1] != "contact_traction"]

        # Add data to the fracture
        not_exported = []
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            # Append aperture
            aperture = self.aperture([sd])
            data.append((sd, "aperture", self.equation_system.evaluate(aperture)))

            # Append fracture gap
            gap = self.fracture_gap([sd])
            data.append((sd, "gap", self.equation_system.evaluate(gap)))

            # Append scaled contact traction
            scaled_contact_traction = self.characteristic_contact_traction(
                [sd]
            ) * self.contact_traction([sd])
            scaled_contact_traction_n = (
                self.normal_component([sd]) @ scaled_contact_traction
            )
            scaled_contact_traction_t = (
                self.tangential_component([sd]) @ scaled_contact_traction
            )
            data.append(
                (
                    sd,
                    "contact_traction",
                    self.units.convert_units(1, "Pa^-1")
                    * self.equation_system.evaluate(scaled_contact_traction),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_n",
                    self.units.convert_units(1, "Pa^-1")
                    * self.equation_system.evaluate(scaled_contact_traction_n),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_t",
                    self.units.convert_units(1, "Pa^-1")
                    * self.equation_system.evaluate(scaled_contact_traction_t),
                )
            )

            # Append slip tendency
            f_norm = pp.ad.Function(
                partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
            )
            mu = self.friction_coefficient([sd])
            scaled_contact_traction_t_norm = f_norm(scaled_contact_traction_t)
            slip_tendency = scaled_contact_traction_t_norm / (
                -scaled_contact_traction_n * mu
            )
            data.append(
                (
                    sd,
                    "slip_tendency",
                    np.absolute(self.equation_system.evaluate(slip_tendency)),
                )
            )

            # Append plastic displacement jump
            plastic_jump = self.plastic_displacement_jump([sd])
            tangential_plastic_jump = self.tangential_component([sd]) @ plastic_jump
            tangential_plastic_jump_increment = pp.ad.time_increment(
                tangential_plastic_jump
            )
            data.append(
                (
                    sd,
                    "tangential_plastic_jump",
                    self.equation_system.evaluate(tangential_plastic_jump),
                )
            )
            data.append(
                (
                    sd,
                    "tangential_plastic_jump_increment",
                    self.equation_system.evaluate(tangential_plastic_jump_increment),
                )
            )

            # Append effective fracture opening
            jump = self.displacement_jump([sd])
            opening = self.normal_component([sd]) @ jump - self.fracture_gap([sd])
            data.append(
                (
                    sd,
                    "opening",
                    self.equation_system.evaluate(opening),
                )
            )

        # Deviation from reference state (displacement).
        if hasattr(self, "has_reference_momentum_state"):
            for i, sd in enumerate(self.mdg.subdomains(dim=self.nd)):
                reference_displacement = pp.ad.TimeDependentDenseArray(
                    "reference_displacement", [sd]
                )
                displacement_deviation = (
                    self.displacement([sd]) - reference_displacement
                )
                displacement_time_increment = pp.ad.time_increment(
                    self.displacement([sd])
                )
                data.append(
                    (
                        sd,
                        "deviation_displacement",
                        self.equation_system.evaluate(displacement_deviation),
                    )
                )
                data.append(
                    (
                        sd,
                        "reference_displacement",
                        self.equation_system.evaluate(reference_displacement),
                    )
                )
                data.append(
                    (
                        sd,
                        "displacement_time_increment",
                        self.equation_system.evaluate(displacement_time_increment),
                    )
                )

        # Deviation from reference state (pressure).
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
                            self.equation_system.evaluate(pressure_deviation),
                            "Pa",
                        ),
                    )
                )
                data.append(
                    (
                        sd,
                        "reference_pressure",
                        self.units.convert_units(
                            self.equation_system.evaluate(reference_pressure),
                            "Pa",
                        ),
                    )
                )

        # Deviation from reference state (interface displacement).
        if hasattr(self, "has_reference_momentum_state"):
            for intf in self.mdg.interfaces(dim=self.nd - 1):
                reference_interface_displacement = pp.ad.TimeDependentDenseArray(
                    "reference_interface_displacement", [intf]
                )
                interface_displacement_deviation = (
                    self.interface_displacement([intf])
                    - reference_interface_displacement
                )
                interface_displacement_time_increment = pp.ad.time_increment(
                    self.interface_displacement([intf])
                )
                data.append(
                    (
                        intf,
                        "deviation_interface_displacement",
                        self.equation_system.evaluate(interface_displacement_deviation),
                    )
                )
                data.append(
                    (
                        intf,
                        "reference_interface_displacement",
                        self.equation_system.evaluate(reference_interface_displacement),
                    )
                )
                data.append(
                    (
                        intf,
                        "interface_displacement_time_increment",
                        self.equation_system.evaluate(
                            interface_displacement_time_increment
                        ),
                    )
                )

        # Check which data could not be exported
        if len(not_exported) > 0:
            not_exported = list(set(not_exported))
            logger.info(f"Not all data could be exported. Missing: {not_exported}")

        # Add contact states
        states = self.compute_fracture_states(concatenate=False)
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            data.append((sd, "states", states[i]))

        # For debugging initialization
        for sd in self.mdg.subdomains():
            try:
                porosity = self.porosity([sd])
                data.append((sd, "porosity", self.equation_system.evaluate(porosity)))
            except Exception:
                not_exported.append("porosity")

        return data


class IterationExporting:
    """Class for exporting iteration-dependent approximations."""

    nonlinear_solver_statistics: pp.SolverStatistics
    """Solver statistics object for the non-linear solver."""

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

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution
        to iterate subdictionary.

        """
        super().after_nonlinear_iteration(nonlinear_increment)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()

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
        n = self.params.get("nl_max_iterations", 10)
        r = 10
        while r <= n:
            r *= 10
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iteration
            + r * self.time_manager.time_index,
        )
        self.nonlinear_solver_statistics.save()

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # Monitor which data could not be exported
        not_exported = []

        # Initialize data list
        data = []

        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
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

            # Append increments if available (zero for first iteration)
            try:
                prev_scaled_values = self.equation_system.get_variable_values(
                    variables=[var], iterate_index=1
                )
            except Exception:
                prev_scaled_values = scaled_values
            inc_values = self.units.convert_units(
                scaled_values - prev_scaled_values, units, to_si=True
            )
            data.append((var.domain, var.name + "_inc", inc_values))

        ## Add residuals for each subproblem.
        # _, residual = self.linear_system
        # equation_blocks = {
        #    name: (
        #        self.equation_system.assembled_equation_indices[name],
        #        list(
        #            self.equation_system._equation_image_space_composition[name].keys()
        #        )[0],
        #        self.equation_system._equation_image_size_info[name]["cells"],
        #    )
        #    for name in self.equation_system._equations
        # }
        # for name, (indices, sd, eq_dim) in equation_blocks.items():
        #    data.append(
        #        (sd, name, residual[indices].reshape((eq_dim, -1), order="F")[0])
        #    )

        # Exclude contact_traction from data and scale it.
        data = [d for d in data if d[1] != "contact_traction"]

        # Add data to the fracture
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            nd_vec_to_normal = self.normal_component([sd])
            nd_vec_to_tangential = self.tangential_component([sd])

            # Append scaled contact traction
            scaled_contact_traction = self.characteristic_contact_traction(
                [sd]
            ) * self.contact_traction([sd])
            t_n: pp.ad.Operator = nd_vec_to_normal @ scaled_contact_traction
            t_t: pp.ad.Operator = nd_vec_to_tangential @ scaled_contact_traction
            data.append(
                (
                    sd,
                    "contact_traction",
                    self.equation_system.evaluate(scaled_contact_traction),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_n",
                    self.units.convert_units(1, "Pa^-1")
                    * self.equation_system.evaluate(t_n),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_t",
                    self.units.convert_units(1, "Pa^-1")
                    * self.equation_system.evaluate(t_t),
                )
            )

            # Append slip tendency
            f_norm = pp.ad.Function(
                partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
            )
            t_t_norm: pp.ad.Operator = f_norm(t_t)
            slip_tendency: pp.ad.Operator = t_t_norm / t_n
            data.append(
                (
                    sd,
                    "slip_tendency",
                    self.equation_system.evaluate(slip_tendency),
                )
            )

            # Append normal and tangential displacement and increments
            u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump([sd])
            u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump([sd])
            u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
            data.append(
                (
                    sd,
                    "u_n",
                    self.equation_system.evaluate(u_n),
                )
            )
            data.append(
                (
                    sd,
                    "u_t",
                    self.equation_system.evaluate(u_t),
                )
            )
            data.append(
                (
                    sd,
                    "u_t_increment",
                    self.equation_system.evaluate(u_t_increment),
                )
            )

            # Append aperture
            aperture = self.aperture([sd])
            data.append((sd, "aperture", self.equation_system.evaluate(aperture)))

            # Append fracture gap
            fracture_gap = self.fracture_gap([sd])
            data.append(
                (sd, "fracture_gap", self.equation_system.evaluate(fracture_gap))
            )

            # Append permeability
            try:
                perm = self.permeability([sd])
                data.append((sd, "perm", self.equation_system.evaluate(perm)))
            except Exception:
                not_exported.append("permeability")

            # Append yield criterion
            try:
                yield_criterion = self.yield_criterion([sd])
                data.append(
                    (
                        sd,
                        "yield_criterion",
                        self.equation_system.evaluate(yield_criterion),
                    )
                )
            except Exception:
                not_exported.append("yield criterion")

            # Append orthogonality
            try:
                orthogonality = self.orthogonality([sd])
                data.append(
                    (
                        sd,
                        "orthogonality",
                        self.equation_system.evaluate(orthogonality),
                    )
                )
            except Exception:
                not_exported.append("orthogonality")

            # Append alignment
            try:
                alignment = self.alignment([sd])
                data.append((sd, "alignment", self.equation_system.evaluate(alignment)))
            except Exception:
                not_exported.append("alignment")

            # Append colinearity condition
            try:
                colinearity_condition = self.colinearity_condition([sd])
                data.append(
                    (
                        sd,
                        "colinearity_condition",
                        self.equation_system.evaluate(colinearity_condition),
                    )
                )
            except Exception:
                not_exported.append("colinearity_condition")

            # Append characteristic of the origin
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
                        self.equation_system.evaluate(characteristic_origin),
                    )
                )
            except Exception:
                not_exported.append("characteristic_origin")

        # Add contact states
        try:
            states = self.compute_fracture_states(concatenate=False)
            for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
                data.append((sd, "contact states", states[i]))
        except Exception:
            not_exported.append("contact states")

        not_exported = list(set(not_exported))
        if len(not_exported) > 0:
            logger.warning(f"Not all data could be exported. Missing: {not_exported}")

        return data
