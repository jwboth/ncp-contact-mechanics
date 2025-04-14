from pathlib import Path
from deepdiff import DeepDiff
import json
from main import generate_case_name
import meshio
from icecream import ic
import argparse

argparser = argparse.ArgumentParser(description="Run single fracture test cases.")
argparser.add_argument(
    "-verbose",
    action="store_true",
    help="Print detailed information about the test results.",
)
args = argparser.parse_args()


# Test different formulations
formulations = [
    ("picard", "rr-nonlinear", 0, "none"),
    ("picard", "rr-linear", 0, "none"),
    ("picard", "ncp-min", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-full", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-min-scaled", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-scaled", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-full-scaled", 0, "origin_and_stick_slip_transition"),
    ("newton", "rr-nonlinear", 0, "none"),
    ("newton", "rr-linear", 0, "none"),
    ("newton", "ncp-min", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-fb-full", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-min-scaled", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-fb-scaled", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-fb-full-scaled", 0, "origin_and_stick_slip_transition"),
]
study = 1
seed = 4
mesh_size = 50
passed = []
not_passed = []
failure_overview = {}

for formulation in formulations:
    # Run the simulation with the specified formulation
    print(f"Testing formulation: {formulation}")
    ad_mode, mode, aa, regularization = formulation

    # Fetch the solver statistics
    folder = generate_case_name(
        study=study,
        seed=seed,
        mesh_size=mesh_size,
        dil=0.1,
        cn=1.0,
        ct=1.0,
        tol=1e-10,
        aa=0,
        regularization=regularization,
        method=mode,
        ad_mode=ad_mode,
        linear_solver="scipy_sparse",
        no_intersections=True,
        no_intersections_angle_cutoff=0.0,
        resolved_intersections=False,
        nice_geometry=False,
        regularized_start=False,
        unitary_units=True,
    )
    folder = Path(folder)
    solver_statistics_filename = (
        Path("visualization") / folder / "solver_statistics.json"
    )
    final_solution_filename = {
        "data_1": Path("visualization") / folder / "data_1_000003.vtu",
        "data_2": Path("visualization") / folder / "data_2_000003.vtu",
        "mortar_1": Path("visualization") / folder / "data_mortar_1_000003.vtu",
    }
    reference_statistics_filename = (
        Path("reference") / folder.parent / ad_mode / "solver_statistics.json"
    )
    reference_solution_filename = {
        "data_1": Path("reference") / folder.parent / ad_mode / "data_1_000003.vtu",
        "data_2": Path("reference") / folder.parent / ad_mode / "data_2_000003.vtu",
        "mortar_1": Path("reference")
        / folder.parent
        / ad_mode
        / "data_mortar_1_000003.vtu",
    }

    # Initiate status
    status = True
    failure = []

    # Check if the files exist
    files_exist = solver_statistics_filename.exists()
    for key in final_solution_filename.keys():
        if not final_solution_filename[key].exists():
            files_exist = False
            break
    if not files_exist:
        failure.append("File not found")

    if files_exist:
        # Compare the final solution files
        diff = {}
        for key in final_solution_filename.keys():
            solution_data = meshio.read(final_solution_filename[key])
            reference_data = meshio.read(reference_solution_filename[key])
            diff[key] = DeepDiff(
                solution_data.__dict__,
                reference_data.__dict__,
                significant_digits=3,
                number_format_notation="e",
                ignore_order=True,
                ignore_numeric_type_changes=True,
            )

    if files_exist and mode == "ncp-min-scaled":
        # Compare the solver statistics in terms of number of iterations
        # - NOTE this comparison only makes sense when using the same method!
        solver_statistics = json.loads(solver_statistics_filename.read_text())
        reference_statistics = json.loads(reference_statistics_filename.read_text())
        for time_index in ["1", "2", "3"]:
            if not (
                solver_statistics[time_index]["status"]
                == reference_statistics[time_index]["status"]
            ):
                failure.append(f"Solver status mismatch at time index {time_index}")
            if not (
                solver_statistics[time_index]["num_iteration"]
                == reference_statistics[time_index]["num_iteration"]
            ):
                failure.append(f"Solver iterations mismatch at time index {time_index}")
            if not (
                solver_statistics[time_index]["residual_norms"]
                == reference_statistics[time_index]["residual_norms"]
            ):
                failure.append(
                    f"Solver residual norms mismatch at time index {time_index}"
                )

    if files_exist:
        for key in final_solution_filename.keys():
            if diff[key] != {}:
                failure.append(f"Formulation {formulation} failed for {key}")
                if args.verbose:
                    print(f"Diff for {key}:")
                    print(diff[key])

    if failure == []:
        passed.append(formulation)
    else:
        not_passed.append(formulation)
        failure_overview[formulation] = failure

# Print the results
ic(passed)
ic(not_passed)
ic(failure_overview)
