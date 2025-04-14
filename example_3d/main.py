"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path
import numpy as np

import porepy as pp
from porepy.numerics.nonlinear import line_search
from setups.geometry import BedrettoGeometry
from setups.physics import (
    Physics,
    fluid_parameters,
    numerics_parameters,
    solid_parameters,
    injection_schedule,
)
import ncp

# from common.newton_return_map import NewtonReturnMap
# from FTHM_Solver.hm_solver import IterativeHMSolver


# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NonlinearRadialReturnModel(
    BedrettoGeometry,  # Geometry
    ncp.ScaledContact,  # Characteristic scalings
    ncp.AuxiliaryContact,  # Yield function, orthognality, and alignment
    ncp.FractureStates,  # Physics based contact states for output only
    ncp.IterationExporting,  # Tailored export
    ncp.LebesgueConvergenceMetrics,  # Convergence metrics
    ncp.LogPerformanceDataVectorial,  # Tailored convergence checks
    ncp.ReverseElasticModuli,  # Characteristic displacement from traction
    Physics,  # Basic model, BC and IC
):
    """Simple Bedretto model solved with Huebers nonlinear radial return formulation."""


# Alart and Curnier formulation, but with scaled contact conditions
class LinearRadialReturnModel(
    ncp.LinearRadialReturnTangentialContact, NonlinearRadialReturnModel
): ...


# NCP Formulations
class NCPModel(
    ncp.NCPNormalContact,
    ncp.NCPTangentialContact2d,
    NonlinearRadialReturnModel,
): ...


# Unscaled variants
class UnscaledNonlinearRadialReturnModel(
    ncp.UnscaledContact, NonlinearRadialReturnModel
): ...


class UnscaledLinearRadialReturnModel(ncp.UnscaledContact, LinearRadialReturnModel): ...


class UnscaledNCPModel(ncp.UnscaledContact, NCPModel): ...


def generate_case_name(
    apply_horizontal_stress,
    num_fractures,
    formulation,
    linearization,
    relaxation,
    linear_solver,
    mass_unit,
):
    folder = Path(
        f"simple_bedretto_{num_fractures}"
        + ("_sigma_h" if apply_horizontal_stress else "")
    )
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_{linear_solver.lower()}"
    if not np.isclose(mass_unit, 1e0):
        name += f"_{mass_unit:.0e}"
    return folder / name


if __name__ == "__main__":
    # Monitor the time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run simple Bedretto case.")
    parser.add_argument(
        "--formulation",
        type=str,
        default="rr-nonlinear",
        help="""Nonlinear formulation to use (rr-nonlinear [default], """
        """rr-linear, ncp-min, ncp-fb).""",
    )
    parser.add_argument(
        "--linearization",
        type=str,
        default="picard",
        help="AD mode (Picard [default], Newton).",
    )
    parser.add_argument(
        "--relaxation",
        type=str,
        default="None",
        help="Relaxation method (None [default]).",
    )
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="scipy_sparse",
        help="Linear solver to use. (scipy_sparse [default], pypardiso).",
    )
    parser.add_argument(
        "--num-fractures",
        type=int,
        default=6,
        help="Number of fractures (1-6 [default]).",
    )
    parser.add_argument(
        "--apply-horizontal-stress",
        action="store_true",
        help="Apply horizontal stress.",
    )
    parser.add_argument(
        "--mass-unit",
        type=float,
        default=1e0,
        help="Mass unit (1e0 [default], 1e6, 1e14).",
    )
    parser.add_argument(
        "--asci-export",
        action="store_true",
        help="Export results in ascii format.",
    )
    args = parser.parse_args()

    # Model parameters
    model_params = {
        # Geometry
        "gmsh_file_name": "msh/gmsh_frac_file.msh",
        "num_fractures": args.num_fractures,
        # Time
        "time_manager": pp.TimeManager(
            # schedule=[0, 2 * pp.DAY] + [(3 + i) * pp.DAY for i in range(5)],
            schedule=[0, 2 * pp.DAY] + [(3 + i) * pp.DAY for i in range(1)],
            dt_init=pp.DAY,
            constant_dt=True,
        ),
        # Material
        "material_constants": {
            "solid": pp.SolidConstants(**solid_parameters),
            "fluid": pp.FluidComponent(**fluid_parameters),
            "numerical": pp.NumericalConstants(**numerics_parameters),
        },
        "apply_horizontal_stress": args.apply_horizontal_stress,
        # User-defined units
        "units": pp.Units(kg=args.mass_unit, m=1, s=1, rad=1),
        # Numerics
        "solver_statistics_file_name": "solver_statistics.json",
        "export_constants_separately": False,
        "linear_solver": "scipy_sparse",
        "max_iterations": 200,  # Needed for export
        "folder_name": Path("visualization")
        / generate_case_name(
            args.apply_horizontal_stress,
            args.num_fractures,
            args.formulation,
            args.linearization,
            args.relaxation,
            args.linear_solver,
            args.mass_unit,
        ),
        "nonlinear_solver_statistics": ncp.AdvancedSolverStatistics,
    }

    # Update the numerical parameter if unscaled formulation is used
    if args.formulation.lower() in [
        "rr-nonlinear-unscaled",
        "rr-linear-unscaled",
        "ncp-min-unscaled",
        "ncp-fb-partial-unscaled",
        "ncp-fb-unscaled",
    ]:
        from dataclasses import dataclass

        @dataclass(kw_only=True, eq=False)
        class UnscaledNumericalConstants(pp.Constants):
            """Data class containing numerical method parameters,
            including characteristic sizes.

            """

            SI_units = dict(
                {
                    "characteristic_displacement": "-",
                    "characteristic_contact_traction": "-",
                    "open_state_tolerance": "Pa",
                    "contact_mechanics_scaling": "Pa",
                }
            )

            characteristic_contact_traction: float = 1.0
            """Characteristic traction used for scaling of contact mechanics."""

            characteristic_displacement: float = 1.0
            """Characteristic displacement used for scaling of contact mechanics."""

            contact_mechanics_scaling: float = 1e0
            """Safety scaling factor, making fractures softer than the matrix."""

            open_state_tolerance: float = 1e-10
            """Tolerance parameter for the tangential characteristic contact mechanics."""

        updated_numerics_parameters = numerics_parameters.copy()
        updated_numerics_parameters.update(
            {
                "open_state_tolerance": 1e-10
                * injection_schedule["reference_pressure"],
                "characteristic_contact_traction": 1.0,
            }
        )
        model_params["material_constants"]["numerical"] = UnscaledNumericalConstants(
            **updated_numerics_parameters
        )

    # Solver parameters
    solver_params = {
        "nonlinear_solver": ncp.AANewtonSolver,  # pp.NewtonSolver,
        "aa_depth": 0,  # Triggers stopping simulation if cycling
        "max_iterations": 200,
        "nl_convergence_tol": 1e-6,
        "nl_convergence_tol_rel": 1e-6,
        "nl_convergence_tol_res": 1e-6,
        "nl_convergence_tol_res_rel": 1e-6,
        "nl_convergence_tol_tight": 1e-10,
        "nl_convergence_tol_rel_tight": 1e-10,
        "nl_convergence_tol_res_tight": 1e-10,
        "nl_convergence_tol_res_rel_tight": 1e-10,
    }

    # Model setup
    Path(model_params["folder_name"]).mkdir(parents=True, exist_ok=True)
    logger.info(f"\n\nRunning {model_params['folder_name']}")

    # Define formulation
    match args.formulation.lower():
        case "rr-nonlinear":
            Model = NonlinearRadialReturnModel

        case "rr-linear":
            Model = LinearRadialReturnModel

        case "ncp-min":
            model_params["ncp_type"] = "min"
            model_params["stick_slip_regularization"] = (
                "origin_and_stick_slip_transition"
            )
            Model = NCPModel

        case "ncp-fb":
            model_params["ncp_type"] = "fb"
            model_params["stick_slip_regularization"] = (
                "origin_and_stick_slip_transition"
            )
            Model = NCPModel

        case "ncp-fb-partial":
            model_params["ncp_type"] = "fb-partial"
            model_params["stick_slip_regularization"] = (
                "origin_and_stick_slip_transition"
            )
            Model = NCPModel

        case "rr-nonlinear-unscaled":
            Model = UnscaledNonlinearRadialReturnModel

        case "rr-linear-unscaled":
            Model = UnscaledLinearRadialReturnModel

        case "ncp-min-unscaled":
            model_params["ncp_type"] = "min"
            model_params["stick_slip_regularization"] = (
                "origin_and_stick_slip_transition"
            )
            Model = UnscaledNCPModel

        case "ncp-fb-unscaled":
            model_params["ncp_type"] = "fb"
            model_params["stick_slip_regularization"] = (
                "origin_and_stick_slip_transition"
            )
            Model = UnscaledNCPModel

        case "ncp-fb-partial-unscaled":
            model_params["ncp_type"] = "fb-partial"
            model_params["stick_slip_regularization"] = (
                "origin_and_stick_slip_transition"
            )
            Model = UnscaledNCPModel

        case _:
            raise ValueError(f"formulation {args.formulation} not recognized.")

    # Choose nonlinear solver (Newton with relaxation)
    match args.linearization.lower():
        case "picard":
            ...

        case "newton":

            class Model(ncp.DarcysLawAd, Model):
                """Enhance with AD of permeability."""

        case _:
            raise ValueError(f"AD mode {args.linearization} not recognized.")

    # Choose relaxation method
    match args.relaxation.lower():
        case "none":
            ...

        case "linesearch":
            raise NotImplementedError(
                "Line search is not implemented yet. Use Picard or Newton."
            )

            class Model(
                pp.models.solution_strategy.ContactIndicators,
                Model,
            ):
                """Added contact indicators for line search."""

            class ConstraintLineSearchNonlinearSolver(
                line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
                line_search.SplineInterpolationLineSearch,  # Technical implementation of
                # the actual search along given update direction
                line_search.LineSearchNewtonSolver,  # General line search.
            ): ...

            solver_params["nonlinear_solver"] = ConstraintLineSearchNonlinearSolver
            solver_params["Global_line_search"] = (
                0  # Set to 1 to use turn on a residual-based line search
            )
            solver_params["Local_line_search"] = (
                1  # Set to 0 to use turn off the tailored line search
            )
            solver_params["adaptive_indicator_scaling"] = (
                1  # Scale the indicator adaptively to increase robustness
            )

        case "return-map":
            raise NotImplementedError(
                "Return map is not implemented yet. Use Picard or Newton."
            )

            class Model(
                NewtonReturnMap,
                Model,
            ):
                """Add return map before each iteration."""

        case _:
            raise ValueError(f"Relaxation method {args.relaxation} not recognized.")

    # Choose linear solver
    match args.linear_solver.lower():
        case "scipy_sparse":
            # Use scipy sparse solver
            model_params["linear_solver"] = "scipy_sparse"
            solver_params["linear_solver"] = "scipy_sparse"
        case "pypardiso":
            # Use pypardiso solver
            model_params["linear_solver"] = "pypardiso"
            solver_params["linear_solver"] = "pypardiso"
        case "fthm":
            raise NotImplementedError(
                "FTHM solver is not implemented yet. Use scipy_sparse or pypardiso."
            )

            class Model(
                IterativeHMSolver,
                Model,
            ): ...

            model_params["linear_solver_config"] = {
                # Avaliable options for THM: CPR, SAMG, FGMRES (fastest to slowest).
                # For HM, this parameter is ignored.
                "solver": "CPR",
                "ksp_monitor": True,  # Enable to see convergence messages from PETSc.
                "logging": False,  # Does not work well with a progress bar.
                "treat_singularity_contact": True,
            }
            solver_params["linear_solver_config"] = model_params["linear_solver_config"]

        case _:
            raise ValueError(f"Linear solver {args.linear_solver} not recognized.")

    # Choose ascii export
    if args.asci_export:

        class Model(
            ncp.ASCIExport,
            Model,
        ):
            """Add ascii export."""

    # Run the model
    model = Model(model_params)
    pp.run_time_dependent_model(model, solver_params)

    # Simple statistics
    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
