from pathlib import Path
from deepdiff import DeepDiff
import json
from main import generate_case_name
import meshio
from icecream import ic
import argparse
import datetime
import subprocess

argparser = argparse.ArgumentParser(description="Run single fracture test cases.")
argparser.add_argument(
    "-verbose",
    action="store_true",
    help="Print detailed information about the test results.",
)
args = argparser.parse_args()


# Test different formulations
linearizations = ["picard", "newton"]
formulations = [
    ("rr-nonlinear", "none"),
    ("rr-linear", "none"),
    ("ncp-min", "origin_and_stick_slip_transition"),
    ("ncp-fb", "origin_and_stick_slip_transition"),
    ("ncp-fb-partial", "origin_and_stick_slip_transition"),
    ("rr-nonlinear-unscaled", "none"),
    ("rr-linear-unscaled", "none"),
    ("ncp-min-unscaled", "origin_and_stick_slip_transition"),
    ("ncp-fb-partial-unscaled", "origin_and_stick_slip_transition"),
    ("ncp-fb-unscaled", "origin_and_stick_slip_transition"),
]
study = 1
seed = 4
mesh_size = 50
mass_unit = 1

passed = []
not_passed = []
failure_overview = {}
performance_passed = []
performance_not_passed = []
performance_failure_overview = {}

for linearization in linearizations:
    for formulation, regularization in formulations:
        combination = (linearization, formulation, regularization)

        # Fetch the solver statistics
        folder = generate_case_name(
            study=study,
            seed=seed,
            mesh_size=mesh_size,
            dil=0.1,
            cn=1.0,
            ct=1.0,
            tol=1e-10,
            regularization=regularization,
            formulation=formulation,
            linearization=linearization,
            linear_solver="scipy_sparse",
            no_intersections=True,
            no_intersections_angle_cutoff=0.0,
            mass_unit=mass_unit,
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
            Path("reference") / folder.parent / linearization / "solver_statistics.json"
        )
        reference_solution_filename = {
            "data_1": Path("reference")
            / folder.parent
            / linearization
            / "data_1_000003.vtu",
            "data_2": Path("reference")
            / folder.parent
            / linearization
            / "data_2_000003.vtu",
            "mortar_1": Path("reference")
            / folder.parent
            / linearization
            / "data_mortar_1_000003.vtu",
        }

        # Initiate status
        status = True
        failure = []
        performance_failure = []

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

        if files_exist:
            for key in final_solution_filename.keys():
                if diff[key] != {}:
                    failure.append(f"Formulation {formulation} failed for {key}")
                    if args.verbose:
                        print(f"Diff for {key}:")
                        print(diff[key])

        if files_exist and formulation == "ncp-min-scaled":
            # Compare the solver statistics in terms of number of iterations
            # - NOTE this comparison only makes sense when using the same method!
            solver_statistics = json.loads(solver_statistics_filename.read_text())
            reference_statistics = json.loads(reference_statistics_filename.read_text())
            for time_index in ["1", "2", "3"]:
                if not (
                    solver_statistics[time_index]["status"]
                    == reference_statistics[time_index]["status"]
                ):
                    performance_failure.append(
                        f"Solver status mismatch at time index {time_index} ({solver_statistics[time_index]['status']} vs {reference_statistics[time_index]['status']})"
                    )
                if not (
                    solver_statistics[time_index]["num_iteration"]
                    == reference_statistics[time_index]["num_iteration"]
                ):
                    performance_failure.append(
                        f"Solver iterations mismatch at time index {time_index} ({solver_statistics[time_index]['num_iteration']} vs {reference_statistics[time_index]['num_iteration']})"
                    )
                if not (
                    solver_statistics[time_index]["residual_norms"]
                    == reference_statistics[time_index]["residual_norms"]
                ):
                    performance_failure.append(
                        f"Solver residual norms mismatch at time index {time_index} ({solver_statistics[time_index]['residual_norms']} vs {reference_statistics[time_index]['residual_norms']})"
                    )

        if failure == []:
            passed.append(combination)
            print(f"Testing formulation: {combination} passed")
        else:
            not_passed.append(combination)
            failure_overview[combination] = failure
            print(f"Testing formulation: {combination} failed")

        if formulation == "ncp-min-scaled":
            if performance_failure == []:
                performance_passed.append(combination)
                print(f"Testing performance: {combination} passed")
            else:
                performance_not_passed.append(combination)
                performance_failure_overview[combination] = performance_failure
                print(f"Testing performance: {combination} failed")

# Print the results
ic(passed)
ic(not_passed)
ic(failure_overview)
ic(performance_passed)
ic(performance_not_passed)
ic(performance_failure_overview)


# Fetch latest git log message
def get_latest_git_commit_message():
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%H %B"], capture_output=True, text=True
    )
    return result.stdout.strip()


latest_commit_message = get_latest_git_commit_message()

# Report the results in txt file annotated by the date and time - append if the file exists
with open(
    f"test_results_2d.txt",
    "a",
) as f:
    f.write("--------------------------------------------------------\n")
    f.write(f"Test run on {datetime.datetime.now()}\n")
    f.write(f"Latest commit message: {latest_commit_message}\n")
    f.write("Passed:\n")
    for item in passed:
        f.write(f"{item}\n")
    f.write("\nNot Passed:\n")
    for item in not_passed:
        f.write(f"{item}\n")
    f.write("\nFailure Overview:\n")
    for key, value in failure_overview.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")
    f.write("Performance Passed:\n")
    for item in performance_passed:
        f.write(f"{item}\n")
    f.write("\nPerformance Not Passed:\n")
    for item in performance_not_passed:
        f.write(f"{item}\n")
    f.write("\nPerformance Failure Overview:\n")
    for key, value in performance_failure_overview.items():
        f.write(f"{key}: {value}\n")
    f.write("--------------------------------------------------------\n")
