"""Fracture stimulation through injection in 2d."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import porepy as pp
from icecream import ic
from setups.geometry import GeometryFromFile, GeometryFromFile_SingleFracs
from setups.physics import (
    Physics,
    fluid_parameters,
    injection_schedule,
    numerics_parameters,
    solid_parameters,
)

import ncp

# from porepy.numerics.nonlinear import line_search


# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Hueber formulation, but with scaled contact conditions
class NonlinearRadialReturnModel(
    GeometryFromFile,  # Geometry
    ncp.ScaledContact,
    ncp.AuxiliaryContact,  # Yield function, orthognality, and alignment
    ncp.FractureStates,  # Physics based conact states
    ncp.IterationExporting,  # Tailored export
    ncp.LebesgueConvergenceMetrics,  # Convergence metrics
    ncp.LogPerformanceDataVectorial,  # Tailored convergence checks
    ncp.ReverseElasticModuli,  # Characteristic displacement from traction
    Physics,  # Model, BC and IC
): ...


# Alart and Curnier formulation, but with scaled contact conditions
class LinearRadialReturnModel(
    ncp.LinearRadialReturnTangentialContact, NonlinearRadialReturnModel
): ...


# NCP Formulations
class NCPModel(
    ncp.NCPNormalContact,
    ncp.NCPTangentialContact,
    NonlinearRadialReturnModel,
): ...


# Unscaled variants
class UnscaledNonlinearRadialReturnModel(
    ncp.UnscaledContact, NonlinearRadialReturnModel
): ...


class UnscaledLinearRadialReturnModel(ncp.UnscaledContact, LinearRadialReturnModel): ...


class UnscaledNCPModel(ncp.UnscaledContact, NCPModel): ...


def generate_case_name(
    study,
    seed,
    mesh_size,
    dil,
    cn,
    ct,
    tol,
    regularization,
    formulation,
    linearization,
    linear_solver,
    no_intersections,
    no_intersections_angle_cutoff,
    mass_unit,
):
    assert np.isclose(ct, cn), "ct and cn must be equal."
    case_name = f"study_{study}/seed_{seed}/mesh_{mesh_size}/{formulation}_{linearization}_{linear_solver}"
    if no_intersections and not np.isclose(no_intersections_angle_cutoff, 0):
        case_name += "_no_int" + f"_{no_intersections_angle_cutoff}"
    if not regularization == "none":
        case_name += f"_reg_{regularization}"
    if not np.isclose(mass_unit, 1e0):
        case_name += f"_unit_{mass_unit}"
    if not np.isclose(dil, 0.1):
        case_name += f"_dil_{dil}"
    if not np.isclose(cn, 1):
        case_name += f"_cnum_{cn}"
    if not np.isclose(tol, 1e-10):
        case_name += f"_tol_{tol}"
    return case_name


if __name__ == "__main__":
    # Montioring time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run single fracture test cases.")
    parser.add_argument("--study", type=int, default=None, help="Study number.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--num_time_steps", type=int, default=5, help="Number of time steps."
    )
    parser.add_argument(
        "--num_iter", type=int, default=200, help="Number of nonlinear iterations."
    )
    parser.add_argument(
        "--formulation", type=str, default="rr-nonlinear", help="Method to use."
    )
    parser.add_argument("--linearization", type=str, default="picard", help="AD mode.")
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="scipy_sparse",
        help="Linear solver to use. (scipy_sparse [default], pypardiso).",
    )
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance.")
    parser.add_argument("--cn", type=float, default=1e0, help="Cnum")
    parser.add_argument(
        "--ct",
        type=float,
        default=1e0,
        help="Cnum in tangential direction for NCP formulation.",
    )
    parser.add_argument("--regularization", type=str, default="none")
    parser.add_argument(
        "--no_intersections", type=str, default=False, help="No intersections."
    )
    parser.add_argument(
        "--no_intersections_angle_cutoff",
        type=float,
        default=np.pi / 2,
        help="Angle of no intersections.",
    )
    parser.add_argument("--mesh_size", type=float, default=10, help="Mesh size.")
    parser.add_argument(
        "--output", type=str, default="visualization", help="base output folder"
    )
    parser.add_argument(
        "--mass-unit",
        type=float,
        default=1e0,
        help="Mass unit (1e0 [default], 1e6, 1e14).",
    )
    parser.add_argument(
        "--asci-export", action="store_true", help="Export data in ASCII format."
    )
    args = parser.parse_args()

    study = args.study
    num_time_steps = args.num_time_steps
    num_iter = args.num_iter
    formulation = args.formulation
    linearization = args.linearization
    linear_solver = args.linear_solver
    cn = args.cn
    ct = args.ct
    tol = args.tol
    seed = args.seed
    mesh_size = args.mesh_size
    if np.isclose(mesh_size, int(mesh_size)):
        mesh_size = int(mesh_size)
    regularization = args.regularization
    no_intersections = args.no_intersections == "True"
    no_intersections_angle_cutoff = args.no_intersections_angle_cutoff
    mass_unit = args.mass_unit

    # Model parameters
    model_params = {
        # Geometry
        "study": study,
        "seed": seed,
        "fracture_generator_file": f"fracture_generator/random_fractures_{seed}.csv",
        "no_intersections_angle_cutoff": no_intersections_angle_cutoff,
        "cell_size_fracture": args.mesh_size,
        "gmsh_file_name": f"msh/gmsh_frac_file_study_{study}_seed_{seed}_mesh_{mesh_size}_intersections_{no_intersections}_angle_cutoff_{no_intersections_angle_cutoff}.msh",
        # Time
        "time_manager": pp.TimeManager(
            schedule=[0, pp.DAY * num_time_steps / 5.0],
            dt_init=pp.DAY / 5.0,
            constant_dt=True,
        ),
        # Material
        "material_constants": {
            "solid": pp.SolidConstants(**solid_parameters),
            "fluid": pp.FluidComponent(**fluid_parameters),
            # "numerical": pp.NumericalConstants(**numerics_parameters), # NOTE: Use tailored open state tol
        },
        "units": (pp.Units(kg=mass_unit, m=1e0, s=1, rad=1)),
        # Numerics
        "stick_slip_regularization": regularization,
        "solver_statistics_file_name": "solver_statistics.json",
        "export_constants_separately": False,
        "linear_solver": linear_solver,  # Needed for setting up solver
        "max_iterations": num_iter,  # Needed for export
    }

    # Use open state tolerance model parameters according to user input
    characteristic_contact_traction = (
        injection_schedule["reference_pressure"]
        if args.formulation.lower()
        in [
            "rr-nonlinear",
            "rr-linear",
            "ncp-min",
            "ncp-fb-partial",
            "ncp-fb",
        ]
        else 1.0
    )
    open_state_tolerance = (
        tol
        if args.formulation.lower()
        in [
            "rr-nonlinear",
            "rr-linear",
            "ncp-min",
            "ncp-fb-partial",
            "ncp-fb",
        ]
        else tol * injection_schedule["reference_pressure"]
    )
    numerics_parameters.update(
        {
            "open_state_tolerance": open_state_tolerance,
            "contact_mechanics_scaling": cn,
            "characteristic_contact_traction": characteristic_contact_traction,
        }
    )

    model_params["material_constants"]["numerical"] = pp.NumericalConstants(
        **numerics_parameters
    )

    # Case name - used for storing results
    dil = model_params["material_constants"]["solid"].dilation_angle
    case_name = generate_case_name(
        study,
        seed,
        mesh_size,
        dil,
        cn,
        ct,
        tol,
        regularization,
        args.formulation.lower(),
        linearization,
        linear_solver,
        no_intersections,
        no_intersections_angle_cutoff,
        mass_unit,
    )
    model_params["folder_name"] = f"{args.output}/" + case_name
    Path(model_params["folder_name"]).mkdir(parents=True, exist_ok=True)
    model_params["nonlinear_solver_statistics"] = ncp.AdvancedSolverStatistics

    # Solver parameters
    solver_params = {
        "nonlinear_solver": ncp.AANewtonSolver,
        "max_iterations": num_iter,
        "aa_depth": 0,  # No aa
        "nl_convergence_tol": 1e-6,
        "nl_convergence_tol_rel": 1e-6,
        "nl_convergence_tol_res": 1e-6,
        "nl_convergence_tol_res_rel": 1e-6,
        "nl_convergence_tol_tight": 1e-10,
        "nl_convergence_tol_rel_tight": 1e-10,
        "nl_convergence_tol_res_tight": 1e-10,
        "nl_convergence_tol_res_rel_tight": 1e-10,
    }

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

        case "ncp-fb-partial":
            model_params["ncp_type"] = "fb-partial"
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

        case "ncp-fb-partial-unscaled":
            model_params["ncp_type"] = "fb-partial"
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

        case _:
            raise ValueError(f"formulation {args.formulation} not recognized.")

    if no_intersections:

        class Model(GeometryFromFile_SingleFracs, Model): ...

    # Choose nonlinear solver (Newton with relaxation)
    match args.linearization.lower():
        case "picard":
            ...

        case "newton":

            class Model(ncp.DarcysLawAd, Model):
                """Enhance with AD of permeability."""

        case _:
            raise ValueError(f"AD mode {args.linearization} not recognized.")

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
