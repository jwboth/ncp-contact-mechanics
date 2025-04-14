from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np
import porepy as pp

# ! ---- MATERIAL PARAMETERS ----

# From I. Stefansson (2024)
# fluid_parameters: dict[str, float] = {
#    "compressibility": 1e-6,
#    "viscosity": 1e-1,
#    "density": 1.0e0,
# }
#
# solid_parameters: dict[str, float] = {
#    "biot_coefficient": 0.8,
#    "permeability": 1e-10,
#    "normal_permeability": 1e-6,  # 1e-4, # Ivar: 1e-6
#    "porosity": 1.0e-2,
#    "shear_modulus": 2e6,
#    "lame_lambda": 2e6,
#    "residual_aperture": 1e-4,  # 1e-2, # Ivar: 1e-3
#    "density": 1e0,
#    "maximum_elastic_fracture_opening": 0e-3,  # Not used
#    "fracture_normal_stiffness": 1e3,  # Not used
#    "fracture_tangential_stiffness": -1,
#    "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
#    "dilation_angle": 0.1,
#    "friction_coefficient": 1.0,
#    # "open_state_tolerance": 1e-10,  # Numerical method parameter
#    "characteristic_displacement": 1.0,
#    "characteristic_contact_traction": 1.0,
# }
# injection_pressure = 1e3

fluid_parameters: dict[str, float] = {
    "compressibility": 1e-10,
    "viscosity": 1e-3,
    "density": 1.0e0,
}

solid_parameters: dict[str, float] = {
    "biot_coefficient": 0.8,
    "permeability": 1e-14,
    "normal_permeability": 1e-14,  # 1e-4, # Ivar: 1e-6
    "porosity": 1.0e-2,
    "shear_modulus": 1e10,
    "lame_lambda": 1e10,
    "residual_aperture": 1e-3,  # 1e-2, # Ivar: 1e-3
    "density": 1e0,
    "maximum_elastic_fracture_opening": 0e-3,  # Not used
    "fracture_normal_stiffness": 1e3,  # Not used
    "fracture_tangential_stiffness": -1,
    "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
    "dilation_angle": 0.1,
    "friction_coefficient": 1.0,
}
injection_schedule = {
    "time": [pp.DAY * 0.2, pp.DAY * 0.4, pp.DAY * 0.6, pp.DAY * 0.8, pp.DAY],
    "pressure": [1e5, 1e6, 1e7, 2e6, 2e6],
    "reference_pressure": 1e7,
}

numerics_parameters: dict[str, float] = {
    # "open_state_tolerance": 1e-10,  # Numerical method parameter
    "characteristic_displacement": 1.0,
    "characteristic_contact_traction": 1.0,
}


# ## From Zabegaev et al. (2025)
# fluid_parameters: dict[str, float] = {
#     "compressibility": 0.0,
#     "density": 998.2,
#     "viscosity": 1e-1,
#     "density": 1.0e0,
# }
#
# solid_parameters: dict[str, float] = {
#     "shear_modulus": 1.2e10,
#     "lame_lambda": 1.2e10,
#     "dilation_angle": 0.1,
#     "friction_coefficient": 0.577,
#     "residual_aperture": 1e-4,  # 1e-2, # Ivar: 1e-3
#     "normal_permeability": 1e-4,  # 1e-4, # Ivar: 1e-6
#     "permeability": 1e-14,
#     "biot_coefficient": 0.47,
#     "porosity": 1.3e-2,
#     "density": 2600,
#     "maximum_elastic_fracture_opening": 0e-3,  # Not used
#     "fracture_normal_stiffness": 1e3,  # Not used
#     "fracture_tangential_stiffness": -1,
#     "fracture_gap": 1e-4,  # Equals the maximum fracture closure.
#     # "open_state_tolerance": 1e-10,  # Numerical method parameter
#     "characteristic_displacement": 1.0,
#     "characteristic_contact_traction": 1.0,
# }


@dataclass(kw_only=True, eq=False)
class ExtendedNumericalConstants(pp.NumericalConstants):
    contact_mechanics_scaling_t: float

    SI_units: ClassVar[dict[str, str]] = dict(
        {
            "characteristic_displacement": "m",
            "characteristic_contact_traction": "Pa",
            "open_state_tolerance": "-",
            "contact_mechanics_scaling": "-",
            "contact_mechanics_scaling_t": "-",
        }
    )

    @property
    def default_constants(self):
        default_constants = super().default_constants
        default_constants.update({"contact_mechanics_scaling_t": 1.0})
        return default_constants

    def contact_mechanics_scaling_t(self):
        return self.constants["contact_mechanics_scaling_t"]


class ADTime:
    def time(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """AD variant of time .

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for time.

        """
        return pp.ad.TimeDependentDenseArray("time", [self.mdg.subdomains()[0]])

    def update_time_ones(self) -> None:
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="time",
                values=self.units.convert_units(
                    np.array([self.time_manager.time]), "s"
                ),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        self.update_time_ones()


class NonzeroInitialCondition:
    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        super().initial_condition()
        for var in self.equation_system.variables:
            if hasattr(self, "initial_" + var.name):
                values = getattr(self, "initial_" + var.name)([var.domain])
                self.equation_system.set_variable_values(
                    values, [var], iterate_index=0, time_step_index=0
                )

    def initial_pressure(self, sd=None):
        val = self.reference_variable_values.pressure
        if sd is None:
            return val
        else:
            return val * np.ones(sd[0].num_cells)

    def initial_t(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Initial contact traction [Pa].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for initial contact traction.

        """
        # TODO: Important?
        sd = subdomains[0]
        traction_vals = np.zeros((self.nd, sd.num_cells))
        # self.characteristic_traction(subdomains).value(
        #    self.equation_system
        # )
        return traction_vals.ravel("F")


class ZeroDisplacementBC:
    solid: pp.SolidConstants
    nd: int
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.

        Parameters:
            sd: Subdomain grid.

        Returns:
            bc: Boundary condition representation.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd)
        bc_faces = boundary_faces.south + boundary_faces.east + boundary_faces.west
        bc = pp.BoundaryConditionVectorial(sd, bc_faces, "dir")

        # Default internal BC is Neumann. We change to Dirichlet, i.e., the
        # mortar variable represents the displacement on the fracture faces.
        bc.internal_to_dirichlet(sd)

        return bc

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """

        # Default is zero.
        vals = np.zeros((self.nd, boundary_grid.num_cells))
        return vals.ravel("F")


class FlowBC:
    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    is_well: Callable[[pp.Grid], bool]
    fluid: pp.FluidComponent

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        pressure = self.reference_variable_values.pressure * np.ones(
            boundary_grid.num_cells
        )

        return pressure

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")


class PressureConstraintWell:
    def update_time_dependent_ad_arrays(self) -> None:
        """Set current injection pressure."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        current_injection_pressure = np.interp(
            self.time_manager.time,
            injection_schedule["time"],
            injection_schedule["pressure"],
            left=0.0,
        )
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_pressure",
                values=np.array([current_injection_pressure]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        std_eq = super().mass_balance_equation(subdomains)

        # Need to embedd in full domain
        sd_indicator = [np.zeros(sd.num_cells) for sd in subdomains]

        # Pick the only subdomain
        fracture_sds = [sd for sd in subdomains if sd.dim == self.nd - 1]

        # Pick a single fracture
        well_sd = fracture_sds[0]

        for i, sd in enumerate(subdomains):
            if sd == well_sd:
                # Pick the center (hardcoded)
                well_loc = np.array(
                    [
                        self.units.convert_units(1500, "m"),
                        self.units.convert_units(1500, "m"),
                        0,
                    ]
                ).reshape((3, 1))

                well_loc_ind = sd.closest_cell(well_loc)

                sd_indicator[i][well_loc_ind] = 1

        # Characteristic functions
        indicator = np.concatenate(sd_indicator)
        reverse_indicator = 1 - indicator

        current_injection_pressure = pp.ad.TimeDependentDenseArray(
            "current_injection_pressure", [self.mdg.subdomains()[0]]
        )
        constrained_eq = self.pressure(subdomains) - current_injection_pressure

        # pp.ad.Scalar(self.units.convert_units(injection_pressure, "Pa"))

        eq_with_pressure_constraint = (
            pp.ad.DenseArray(reverse_indicator) * std_eq
            + pp.ad.DenseArray(indicator) * constrained_eq
        )
        eq_with_pressure_constraint.set_name(
            "mass_balance_equation_with_constrained_pressure"
        )
        return eq_with_pressure_constraint


class Physics(
    ADTime,
    PressureConstraintWell,
    FlowBC,
    ZeroDisplacementBC,
    NonzeroInitialCondition,
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    pp.poromechanics.Poromechanics,  # Basic model
): ...
