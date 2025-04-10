# TODO: Integrate into NCP after experimenting

import numpy as np
import scipy.sparse as sps
import porepy as pp
from icecream import ic

import logging

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScaledNCPAdapters:
    def characteristic_distance(self, subdomains):
        return self.characteristic_displacement(subdomains)


class AdaptiveDarcysLawAd:  # (pp.constitutive_laws.DarcysLawAd):
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        no_contact_states_changes = self.no_contact_states_change(subdomains).value(
            self.equation_system
        )[0]

        # ic(no_contact_states_changes)
        # print(super().darcy_flux_discretization(subdomains))

        if all([sd.dim < self.nd for sd in subdomains]) and np.isclose(
            no_contact_states_changes, 1
        ):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)


class DarcysLawAd(pp.constitutive_laws.DarcysLawAd):
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)


class ReverseElasticModuli:
    """Same as ElasticModuli, but with reversed assignment of characteristic values."""

    def characteristic_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Characteristic traction [Pa].

        Parameters:
            subdomains: List of subdomains where the characteristic traction is defined.

        Returns:
            Scalar operator representing the characteristic traction.

        """
        t_char = pp.ad.Scalar(self.numerical.characteristic_contact_traction)
        t_char.set_name("characteristic_contact_traction")
        return t_char

    def characteristic_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Characteristic displacement [m].

        Parameters:
            subdomains: List of subdomains where the characteristic displacement is
                defined.

        Returns:
            Scalar operator representing the characteristic displacement.

        """
        size = pp.ad.Scalar(np.max(self.domain.side_lengths()))
        u_char = (
            self.characteristic_contact_traction(subdomains)
            * size
            / self.youngs_modulus(subdomains)
        )
        u_char.set_name("characteristic_displacement")
        return u_char


class ContactStateDetector:
    def no_contact_states_change(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Identifier for whether no contact states change.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            switch: Switch as scalar.

        """
        return pp.ad.TimeDependentDenseArray(
            "no_contact_states_change", [self.mdg.subdomains()[0]]
        )

    def update_time_dependent_ad_arrays(self) -> None:
        """Start with min NCP formulation."""
        super().update_time_dependent_ad_arrays()
        self.update_num_contact_states_changes()

    def update_num_contact_states_changes(self) -> None:
        num_contact_state_changes = (
            self.nonlinear_solver_statistics.total_contact_state_changes
        )
        num_iterations = self.nonlinear_solver_statistics.num_iteration
        no_contact_states_change = num_contact_state_changes < 1 and num_iterations > 1
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="no_contact_states_change",
                values=np.array([int(no_contact_states_change)]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )


class MinFbSwitch:
    def min_fb_switch(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Switch between Fischer-Burmeister and min NCP formulations.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            switch: Switch as scalar.

        """
        return pp.ad.TimeDependentDenseArray("active_min", [self.mdg.subdomains()[0]])

    def update_min_fb_switch(self, active_min: bool) -> None:
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="active_min",
                values=np.array([int(active_min)]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )
        logging.info(f"Switched to min NCP: {active_min}")

    def update_time_dependent_ad_arrays(self) -> None:
        """Start with min NCP formulation."""
        super().update_time_dependent_ad_arrays()
        self.update_min_fb_switch(active_min=True)


class AdaptiveCnum:
    """Allow to adapt the numerical constant for contact mechanics."""

    def cnum(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Numerical constant (variable).

        Parameters:
            subdomains: List of subdomains.

        Returns:
            cnum: Numerical constant as scalar.

        """
        return pp.ad.TimeDependentDenseArray("cnum", [self.mdg.subdomains()[0]])

    def get_cnum(self) -> float:
        """Get the current cnum value."""
        return pp.ad.TimeDependentDenseArray("cnum", [self.mdg.subdomains()[0]]).value(
            self.equation_system
        )[0]

    def update_cnum(self, val: float | None = None) -> None:
        if val is None:
            cnum = self.numerical.contact_mechanics_scaling
        else:
            cnum = val
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="cnum",
                values=np.array([cnum]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )
        logging.info(f"New cnum value: {cnum}")

    def cnum_t(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Numerical constant (variable).

        Parameters:
            subdomains: List of subdomains.

        Returns:
            cnum: Numerical constant as scalar.

        """
        return pp.ad.TimeDependentDenseArray("cnum_t", [self.mdg.subdomains()[0]])

    def get_cnum_t(self) -> float:
        """Get the current cnum value."""
        return pp.ad.TimeDependentDenseArray(
            "cnum_t", [self.mdg.subdomains()[0]]
        ).value(self.equation_system)[0]

    def update_cnum_t(self, val: float | None = None) -> None:
        if val is None:
            cnum_t = self.numerical.contact_mechanics_scaling_t
        else:
            cnum_t = val
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="cnum_t",
                values=np.array([cnum_t]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )
        logging.info(f"New cnum_t value: {cnum_t}")

    def update_time_dependent_ad_arrays(self) -> None:
        """Add update of cnum."""
        super().update_time_dependent_ad_arrays()
        self.update_cnum()
        self.update_cnum_t()


class RegularizedStart:
    def regularization_type(self):
        options = ["none", "u=ex", "u_t=ex", "t=0", "t_t=0", "t_t=half"]
        return options[5]
        return self.params.get("regularization_type", "none")

    def before_nonlinear_loop(self):
        super().before_nonlinear_loop()

        # Fetch current values
        values = self.equation_system.get_variable_values(iterate_index=0)

        if self.regularization_type() == "none":
            # Do nothing (default)
            ...
        elif self.regularization_type() == "u=ex":
            # Extrapolate u from previous time step
            assert (
                np.count_nonzero(
                    [
                        variable.name == "u"
                        for variable in self.equation_system.variables
                    ]
                )
                == 1
            )
            for variable in self.equation_system.variables:
                if variable.name == "u" and hasattr(self, "dt_u"):
                    ic("values reset u=ex")
                    indices = self.equation_system.dofs_of([variable])
                    values[indices] += self.dt_u * self.time_manager.dt
                    break
            self.equation_system.set_variable_values(
                values,
                iterate_index=0,
            )
        elif self.regularization_type() == "u_t=ex" and hasattr(
            self, "u_interface_t_dt"
        ):
            ic("values reset u_t=ex")
            self.equation_system.set_variable_values(
                self.u_interface_t_dt * self.time_manager.dt,
                variables=[
                    var
                    for var in self.equation_system.variables
                    if var.name == "u_interface"
                ],
                iterate_index=0,
                additive=True,
            )
            print(np.linalg.norm(self.u_interface_t_dt))
        elif self.regularization_type() == "t=0":
            ic("values reset t=0")
            for variable in self.equation_system.variables:
                if variable.name == "t":
                    indices = self.equation_system.dofs_of([variable])
                    values[indices] = 0
                    break
            self.equation_system.set_variable_values(
                values,
                iterate_index=0,
            )
        elif self.regularization_type() == "t_t=0" and hasattr(self, "normal_traction"):
            ic("values reset t_t=0")
            self.equation_system.set_variable_values(
                self.normal_traction,
                variables=[
                    var for var in self.equation_system.variables if var.name == "t"
                ],
                iterate_index=0,
            )
        elif self.regularization_type() == "t_t=half":
            values = self.equation_system.get_variable_values(
                variables=[
                    var for var in self.equation_system.variables if var.name == "t"
                ],
                iterate_index=0,
            )
            values *= 0.5
            self.equation_system.set_variable_values(
                values,
                variables=[
                    var for var in self.equation_system.variables if var.name == "t"
                ],
                iterate_index=0,
            )

    def u_jump_t_increment(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        nd_vec_to_tangential = self.tangential_component(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        return u_t_increment

    def after_nonlinear_convergence(self):
        # Matrix-fracture interfaces
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        interfaces = [intf for intf in interfaces if intf.dim == self.nd - 1]
        left_projections = {}
        right_projections = {}
        for intf in interfaces:
            nc = intf.num_cells
            nd = self.nd
            if intf.num_sides() == 1:
                return sps.dia_matrix((np.ones(nc * nd), 0), shape=(nd * nc, nd * nc))
            elif intf.num_sides() == 2:
                # By the ordering of the mortar cells, we know that all cells on the one
                # side are put first, then the other side. Set + and - accordingly.
                from porepy.grids.mortar_grid import MortarSides

                left_side = MortarSides.LEFT_SIDE
                right_side = MortarSides.RIGHT_SIDE
                left_data = np.hstack(
                    (
                        np.ones(intf.side_grids[left_side].num_cells * nd),
                        np.zeros(intf.side_grids[right_side].num_cells * nd),
                    )
                )
                right_data = np.hstack(
                    (
                        np.zeros(intf.side_grids[left_side].num_cells * nd),
                        np.ones(intf.side_grids[right_side].num_cells * nd),
                    )
                )
                left_projections[intf] = sps.dia_matrix(
                    (left_data, 0), shape=(nd * nc, nd * nc)
                )
                right_projections[intf] = sps.dia_matrix(
                    (right_data, 0), shape=(nd * nc, nd * nc)
                )

        # Construct left_project and right_projection by setting together a block matrix
        left_projection = sps.block_diag(
            [left_projections[intf] for intf in interfaces], format="csr"
        )
        right_projection = sps.block_diag(
            [right_projections[intf] for intf in interfaces], format="csr"
        )

        # Interface displacement time derivative
        interface_displacement_inc = (
            pp.ad.time_increment(self.interface_displacement(interfaces))
        ).value(self.equation_system)
        interface_displacement_dt = interface_displacement_inc / self.time_manager.dt

        # Projections
        normal_component = self.normal_component(subdomains).value(self.equation_system)
        tangential_component = self.tangential_component(subdomains).value(
            self.equation_system
        )

        local_coordinates = self.local_coordinates(subdomains).value(
            self.equation_system
        )
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        mortar_to_secondary_avg = mortar_projection.mortar_to_secondary_avg.value(
            self.equation_system
        )
        secondary_to_mortar_avg = mortar_projection.secondary_to_mortar_avg.value(
            self.equation_system
        )

        # Left and right side values of interface displacement increment
        left_u_interface_t_dt = (
            tangential_component
            @ local_coordinates
            @ mortar_to_secondary_avg
            @ left_projection
            @ interface_displacement_dt
        )
        right_u_interface_t_dt = (
            tangential_component
            @ local_coordinates
            @ mortar_to_secondary_avg
            @ right_projection
            @ interface_displacement_dt
        )

        # Store the increment but projected back to the full nd Mortar grid space
        self.u_interface_t_dt = (
            left_projection.T
            @ secondary_to_mortar_avg
            @ local_coordinates.T
            @ tangential_component.T
            @ left_u_interface_t_dt
            + right_projection.T
            @ secondary_to_mortar_avg
            @ local_coordinates.T
            @ tangential_component.T
            @ right_u_interface_t_dt
        )

        # ! ---- ANOTHER APPROACH ----

        # Tangential displacement jump
        self.u_jump_t_inc = (
            self.u_jump_t_increment(self.mdg.subdomains(dim=self.nd - 1)).value(
                self.equation_system
            )
            / self.time_manager.dt
        )

        # Normal traction
        self.normal_traction = (
            normal_component.T
            @ normal_component
            @ self.contact_traction(subdomains).value(self.equation_system)
        )

        # The displacement jmup is expressed in the local coordinates of the fracture.
        # First use the sign of the mortar sides to get a difference, then map first
        # from the interface to the fracture, and finally to the local coordinates.

        # rotated_jumps: pp.ad.Operator = (
        #     self.local_coordinates(subdomains)
        #     @ mortar_projection.mortar_to_secondary_avg
        #     @ self.interface_displacement(interfaces)
        # )
        # rotated_jumps.set_name("Rotated_displacement_jump")

        # print("after", self.u_interface_t_inc.shape, tangential_component.shape)

        # # Store the increment in time step for the displacement
        # values = self.equation_system.get_variable_values(iterate_index=0)
        # values_previous = self.equation_system.get_variable_values(time_step_index=0)
        # for variable in self.equation_system.variables:
        #     indices = self.equation_system.dofs_of([variable])
        #     if variable.name == "u":
        #         print("u")
        #         u_inc = values[indices] - values_previous[indices]
        # self.dt_u = u_inc / self.time_manager.dt

        # Shift values etc.
        super().after_nonlinear_convergence()
