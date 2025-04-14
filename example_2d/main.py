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
    ExtendedNumericalConstants,
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
class ScaledRadialReturnModel(
    ncp.ReverseElasticModuli,  # Characteristic displacement from traction
    GeometryFromFile,  # Geometry
    Physics,  # BC and IC
    ncp.AuxiliaryContact,  # Yield function, orthognality, and alignment
    ncp.FractureStates,  # Physics based conact states
    ncp.IterationExporting,  # Tailored export
    ncp.LebesgueConvergenceMetrics,  # Convergence metrics
    ncp.LogPerformanceDataVectorial,  # Tailored convergence checks
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    pp.poromechanics.Poromechanics,  # Basic model
):
    ...
    """Mixed-dimensional poroelastic problem."""


class ScaledLinearRadialReturnModel(
    ncp.LinearRadialReturnTangentialContact, ScaledRadialReturnModel
): ...


#
#
## Old PorePy formulation of contact without scaling
# class UnscaledRadialReturnModel(
#    ncp.UnscaledContact,
#    ScaledRadialReturnModel,
# ): ...
#
#
# NCP Formulations
class ScaledNCPModel(
    ncp.AdaptiveCnum,
    ncp.MinFbSwitch,
    ncp.ScaledContact,
    ncp.NCPNormalContact,
    ncp.NCPTangentialContact2d,
    ScaledRadialReturnModel,
): ...


class NCPModel(ncp.UnscaledContact, ScaledNCPModel): ...


def generate_case_name(
    study,
    seed,
    mesh_size,
    dil,
    cn,
    ct,
    tol,
    aa,
    regularization,
    method,
    ad_mode,
    linear_solver,
    no_intersections,
    no_intersections_angle_cutoff,
    resolved_intersections,
    nice_geometry,
    regularized_start,
    unitary_units,
):
    assert np.isclose(ct, cn), "ct and cn must be equal."
    case_name = (
        f"study_{study}/seed_{seed}/mesh_{mesh_size}/{method}_{ad_mode}_{linear_solver}"
    )
    if no_intersections and not np.isclose(no_intersections_angle_cutoff, 0):
        case_name += "_no_int" + f"_{no_intersections_angle_cutoff}"
    if not regularization == "none":
        case_name += f"_reg_{regularization}"
    if not unitary_units:
        case_name += "_unitary_F"
    if not np.isclose(dil, 0.1):
        case_name += f"_dil_{dil}"
    if not np.isclose(cn, 1):
        case_name += f"_cnum_{cn}"
    if not np.isclose(aa, 0):
        case_name += f"_aa_{aa}"
    if not np.isclose(tol, 1e-10):
        case_name += f"_tol_{tol}"
    case_name += "_res_int" if resolved_intersections else ""
    case_name += "_nice" if nice_geometry else ""
    case_name += "_reg_start_dtu" if regularized_start else ""
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
    parser.add_argument("--ad-mode", type=str, default="newton", help="AD mode.")
    parser.add_argument("--mode", type=str, default="ncp-min", help="Method to use.")
    parser.add_argument("--linear-solver", type=str, help="Linear solver.")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance.")
    parser.add_argument("--cn", type=float, default=1e0, help="Cnum")
    parser.add_argument(
        "--ct",
        type=float,
        default=1e0,
        help="Cnum in tangential direction for NCP formulation.",
    )
    parser.add_argument("--regularization", type=str, default="none")
    parser.add_argument("--aa", type=int, default=0, help="AA depth.")
    parser.add_argument("--unitary_units", type=str, default="True", help="Units.")
    parser.add_argument(
        "--no_intersections", type=str, default=False, help="No intersections."
    )
    parser.add_argument(
        "--no_intersections_angle_cutoff",
        type=float,
        default=np.pi / 2,
        help="Angle of no intersections.",
    )
    parser.add_argument(
        "--resolved_intersections",
        type=str,
        default=False,
        help="Resolved intersections.",
    )
    parser.add_argument(
        "--nice-geometry",
        type=str,
        default=False,
        help="Use nice geometry for the fracture.",
    )
    parser.add_argument(
        "--regularized_start", type=str, default=False, help="Regularized start."
    )
    parser.add_argument("--mesh_size", type=float, default=10, help="Mesh size.")
    parser.add_argument(
        "--output", type=str, default="visualization", help="base output folder"
    )
    parser.add_argument(
        "--asci-export", action="store_true", help="Export data in ASCII format."
    )
    args = parser.parse_args()

    study = args.study
    num_time_steps = args.num_time_steps
    num_iter = args.num_iter
    ad_mode = args.ad_mode
    mode = args.mode
    linear_solver = args.linear_solver
    cn = args.cn
    ct = args.ct
    tol = args.tol
    aa_depth = args.aa
    seed = args.seed
    mesh_size = args.mesh_size
    if np.isclose(mesh_size, int(mesh_size)):
        mesh_size = int(mesh_size)
    regularization = args.regularization
    unitary_units = args.unitary_units == "True"
    no_intersections = args.no_intersections == "True"
    no_intersections_angle_cutoff = args.no_intersections_angle_cutoff
    resolved_intersections = args.resolved_intersections == "True"
    nice_geometry = args.nice_geometry == "True"
    regularized_start = args.regularized_start == "True"

    # TODO clean up
    assert unitary_units
    assert not resolved_intersections
    assert not nice_geometry
    assert not regularized_start

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
        # TODO rm
        # "units": (
        #     pp.Units(kg=1e0, m=1e0, s=1, rad=1)
        #     if unitary_units
        #     # else pp.Units(kg=1e4, m=1e-2, s=1, rad=1)
        #     else pp.Units(kg=1e6, m=1, s=1, rad=1)
        # ),
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
        if mode
        in [
            "rr-nonlinear",
            "rr-linear",
            "ncp-min-scaled",
            "ncp-fb-scaled",
            "ncp-fb-full-scaled",
        ]
        else 1.0
    )
    open_state_tolerance = (
        tol
        if mode
        in [
            "rr-nonlinear",
            "rr-linear",
            "ncp-min-scaled",
            "ncp-fb-scaled",
            "ncp-fb-full-scaled",
        ]
        else tol * injection_schedule["reference_pressure"]
    )
    numerics_parameters.update(
        {
            "open_state_tolerance": open_state_tolerance,
            "contact_mechanics_scaling": cn,
            "contact_mechanics_scaling_t": ct,
            "characteristic_contact_traction": characteristic_contact_traction,
        }
    )

    model_params["material_constants"]["numerical"] = ExtendedNumericalConstants(
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
        aa_depth,
        regularization,
        mode,
        ad_mode,
        linear_solver,
        no_intersections,
        no_intersections_angle_cutoff,
        resolved_intersections,
        nice_geometry,
        regularized_start,
        unitary_units,
    )
    model_params["folder_name"] = f"{args.output}/" + case_name
    Path(model_params["folder_name"]).mkdir(parents=True, exist_ok=True)
    model_params["nonlinear_solver_statistics"] = ncp.AdvancedSolverStatistics

    # Solver parameters
    solver_params = {
        "nonlinear_solver": ncp.AANewtonSolver,
        "max_iterations": num_iter,
        "aa_depth": aa_depth,
        "nl_convergence_tol": 1e-6,
        "nl_convergence_tol_rel": 1e-6,
        "nl_convergence_tol_res": 1e-6,
        "nl_convergence_tol_res_rel": 1e-6,
        "nl_convergence_tol_tight": 1e-10,
        "nl_convergence_tol_rel_tight": 1e-10,
        "nl_convergence_tol_res_tight": 1e-10,
        "nl_convergence_tol_res_rel_tight": 1e-10,
    }

    if args.asci_export:

        class ScaledRadialReturnModel(ncp.ASCIExport, ScaledRadialReturnModel): ...

        class NCPModel(ncp.ASCIExport, NCPModel): ...

        class ScaledNCPModel(ncp.ASCIExport, ScaledNCPModel): ...

        class ScaledLinearRadialReturnModel(
            ncp.ASCIExport, ScaledLinearRadialReturnModel
        ): ...

    if no_intersections:

        class ScaledNCPModel(GeometryFromFile_SingleFracs, ScaledNCPModel): ...

        class NCPModel(GeometryFromFile_SingleFracs, NCPModel): ...

        class ScaledRadialReturnModel(
            GeometryFromFile_SingleFracs, ScaledRadialReturnModel
        ): ...

        class ScaledLinearRadialReturnModel(
            GeometryFromFile_SingleFracs, ScaledLinearRadialReturnModel
        ): ...

        # class UnscaledRadialReturnModel(
        #    GeometryFromFile_SingleFracs, UnscaledRadialReturnModel
        # ): ...

    match ad_mode:
        case "picard":
            ...

        case "newton":

            class ScaledNCPModel(ncp.DarcysLawAd, ScaledNCPModel): ...

            class NCPModel(ncp.DarcysLawAd, NCPModel): ...

            class ScaledRadialReturnModel(ncp.DarcysLawAd, ScaledRadialReturnModel): ...

            class ScaledLinearRadialReturnModel(
                ncp.DarcysLawAd, ScaledLinearRadialReturnModel
            ): ...

        case "newton_adaptive":

            class ScaledNCPModel(ncp.AdaptiveDarcysLawAd, ScaledNCPModel): ...

            class NCPModel(ncp.AdaptiveDarcysLawAd, NCPModel): ...

            class ScaledRadialReturnModel(
                ncp.AdaptiveDarcysLawAd, ScaledRadialReturnModel
            ): ...

            class ScaledLinearRadialReturnModel(
                ncp.AdaptiveDarcysLawAd, ScaledLinearRadialReturnModel
            ): ...

        case _:
            raise ValueError(f"AD mode {ad_mode} not recognized.")

    # elif resolved_intersections:
    #     # class NCPModel(EGS_ConstrainedResolvedFracs_2d, NCPModel): ...

    #     # class ScaledRadialReturnModel(
    #     #    EGS_ConstrainedResolvedFracs_2d, ScaledRadialReturnModel
    #     # ): ...

    #     # class UnscaledRadialReturnModel(
    #     #    EGS_ConstrainedResolvedFracs_2d, UnscaledRadialReturnModel
    #     # ): ...

    #     class NCPModel(EGS_ResolvedFracs_2d, NCPModel): ...

    #     class ScaledRadialReturnModel(
    #         EGS_ResolvedFracs_2d, ScaledRadialReturnModel
    #     ): ...

    #     class UnscaledRadialReturnModel(
    #         EGS_ResolvedFracs_2d, UnscaledRadialReturnModel
    #     ): ...

    # elif nice_geometry:

    #     class NCPModel(EGS_NiceGeometry_2d, NCPModel): ...

    #     class ScaledRadialReturnModel(EGS_NiceGeometry_2d, ScaledRadialReturnModel): ...

    #     class UnscaledRadialReturnModel(
    #         EGS_NiceGeometry_2d, UnscaledRadialReturnModel
    #     ): ...

    # if regularized_start:

    #     class NCPModel(RegularizedStart, NCPModel): ...

    # Model setup
    logger.info(f"\n\nRunning {case_name}")
    ic(model_params["folder_name"])
    if mode == "rr-nonlinear":
        model = ScaledRadialReturnModel(model_params)

    elif mode == "rr-linear":
        model = ScaledLinearRadialReturnModel(model_params)

    elif mode == "ncp-min":
        model_params["ncp_type"] = "min"
        model = NCPModel(model_params)

    elif mode == "ncp-min-scaled":
        model_params["ncp_type"] = "min"
        model = ScaledNCPModel(model_params)

    elif mode == "ncp-fb":
        model_params["ncp_type"] = "fb"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-star":
        model_params["ncp_type"] = "fb-star"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-scaled":
        model_params["ncp_type"] = "fb"
        model = ScaledNCPModel(model_params)

    elif mode == "ncp-min-linear":
        model_params["ncp_type"] = "min-linear"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-linear":
        model_params["ncp_type"] = "fb-linear"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-full":
        model_params["ncp_type"] = "fb-full"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-full-star":
        model_params["ncp_type"] = "fb-full-star"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-full-scaled":
        model_params["ncp_type"] = "fb-full"
        model = ScaledNCPModel(model_params)

    elif mode == "ncp-min-alternative-stick":
        model_params["ncp_type"] = "min-alternative-stick"
        model = NCPModel(model_params)

    elif mode == "ncp-min-sqrt":
        model_params["ncp_type"] = "min-sqrt"
        model = NCPModel(model_params)

    elif mode == "ncp-min-log":
        model_params["ncp_type"] = "min-log"
        model = NCPModel(model_params)

    elif mode == "ncp-min-exp":
        model_params["ncp_type"] = "min-exp"
        model = NCPModel(model_params)

    elif mode == "ncp-min-log-reg":
        model_params["ncp_type"] = "min-log-reg"
        model = NCPModel(model_params)

    elif mode == "ncp-min-exp-reg":
        model_params["ncp_type"] = "min-exp-reg"
        model = NCPModel(model_params)

    elif mode == "ncp-min-no-intersections":
        model_params["ncp_type"] = "min-no-intersections"
        model = NCPModel(model_params)

    # elif mode == "ncp-min-mu":
    #    model_params["ncp_type"] = "min_mu"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-active-set":
    #    # ut in sticking mode

    #    model_params["ncp_type"] = "min-active-set"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-fb":
    #    # Novel NCP formulation - with Fischer-Burmeister NCP formulation

    #    model_params["ncp_type"] = "min/fb"
    #    solver_params["aa_depth"] = -3
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-fb-rr":
    #    # Novel NCP formulation - with Fischer-Burmeister NCP formulation

    #    model_params["ncp_type"] = "min-fb-rr"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-fb-consistent":
    #    model_params["ncp_type"] = "min-fb-consistent"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-consistent-reg-rr":
    #    model_params["ncp_type"] = "min-consistent-reg-rr"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-sqrt-star":
    #    model_params["ncp_type"] = "min-sqrt-star"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min/rr":
    #    model_params["ncp_type"] = "min/rr"
    #    model = NCPModel(model_params)

    # elif mode == "ncp-min-consistent-scaled":
    #    model_params["ncp_type"] = "min-consistent"
    #    model_params["material_constants"]["solid"]._constants[
    #        "characteristic_displacement"
    #    ] = 1e-2
    #    model = ScaledNCPModel(model_params)

    # elif mode == "ncp-fb-consistent":
    #    model_params["ncp_type"] = "fb-consistent"
    #    model = NCPModel(model_params)

    # elif mode == "rr-linesearch":
    # Need to integrate pp.models.solution_strategy.ContactIndicators in model class
    #    # porepy-main-1.10

    #    model_params["material_constants"]["solid"]._constants[
    #        "characteristic_displacement"
    #    ] = 1e-2
    #    model = ScaledRadialReturnModel(model_params)

    #    class ConstraintLineSearchNonlinearSolver(
    #        line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
    #        line_search.SplineInterpolationLineSearch,  # Technical implementation of the actual search along given update direction
    #        line_search.LineSearchNewtonSolver,  # General line search.
    #    ): ...

    #    solver_params["nonlinear_solver"] = ConstraintLineSearchNonlinearSolver
    #    solver_params["Global_line_search"] = (
    #        0  # Set to 1 to use turn on a residual-based line search
    #    )
    #    solver_params["Local_line_search"] = (
    #        1  # Set to 0 to use turn off the tailored line search
    #    )
    #    solver_params["adaptive_indicator_scaling"] = (
    #        1  # Scale the indicator adaptively to increase robustness
    #    )

    # elif mode == "rr-unscaled":
    #    # porepy-main-1.10 but with unscaled contact conditions like in porepy-main-1.9

    #    model = UnscaledRadialReturnModel(model_params)

    else:
        raise ValueError(f"Mode {mode} not recognized. Choose 'ncp' or 'rr'.")

    pp.run_time_dependent_model(model, solver_params)

    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
