from pathlib import Path
from deepdiff import DeepDiff
import json
from main import generate_case_name
import meshio
import argparse
from icecream import ic
import datetime
import subprocess
import numpy as np

argparser = argparse.ArgumentParser(description="Run single fracture test cases.")
argparser.add_argument(
    "-verbose",
    action="store_true",
    help="Print detailed information about the test results.",
)
args = argparser.parse_args()

# Test different formulations
horizontal_stresses = [True, False]
formulations = [
    "rr-nonlinear",
    "rr-linear",
    "ncp-min",
    "ncp-fb",
    "rr-nonlinear-unscaled",
    "rr-linear-unscaled",
    "ncp-min-unscaled",
    "ncp-fb-unscaled",
]
mass_units = [1, 1e10]
num_fractures = 2
passed = []
not_passed = []
failure_overview = {}
performance_passed = []
performance_not_passed = []
performance_failure_overview = {}

for apply_horizontal_stress in horizontal_stresses:
    for mass_unit in mass_units:
        mass_unit_str = "1" if mass_unit == 1 else "1e+10"
        for formulation in formulations:
            # Run the simulation with the specified formulation
            combination = (apply_horizontal_stress, mass_unit, formulation)

            # Fetch the solver statistics
            folder = generate_case_name(
                apply_horizontal_stress,
                num_fractures,
                formulation,
                "picard",
                "None",
                "scipy_sparse",
                mass_unit,
            )
            solver_statistics_filename = (
                Path("visualization") / folder / "solver_statistics.json"
            )
            final_solution_filename = {
                "data_2": Path("visualization") / folder / "data_2_000003.vtu",
                "data_3": Path("visualization") / folder / "data_3_000003.vtu",
                "mortar_2": Path("visualization") / folder / "data_mortar_2_000003.vtu",
            }

            # Fetch references
            reference_statistics_filename = (
                Path("reference")
                / folder.parent
                / mass_unit_str
                / "solver_statistics.json"
            )
            reference_solution_filename = {
                "data_2": Path("reference") / folder.parent / "1" / "data_2_000003.vtu",
                "data_3": Path("reference") / folder.parent / "1" / "data_3_000003.vtu",
                "mortar_2": Path("reference")
                / folder.parent
                / "1"
                / "data_mortar_2_000003.vtu",
            }

            # Fill in with the missing data_1 and mortar_1 files if the exist
            if (Path("visualization") / folder / "data_1_000000.vtu").exists():
                final_solution_filename["data_1"] = (
                    Path("visualization") / folder / "data_1_000003.vtu"
                )
                reference_solution_filename["data_1"] = (
                    Path("reference")
                    / folder.parent
                    / mass_unit_str
                    / "data_1_000003.vtu"
                )
            if (Path("visualization") / folder / "data_mortar_1_000000.vtu").exists():
                final_solution_filename["mortar_1"] = (
                    Path("visualization") / folder / "data_mortar_1_000003.vtu"
                )
                reference_solution_filename["mortar_1"] = (
                    Path("reference")
                    / folder.parent
                    / mass_unit_str
                    / "data_mortar_1_000003.vtu"
                )

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

                    def custom_compare(x, y, abs_tol=1e-6, rel_tol=1e-1):
                        try:
                            diff = x - y
                        except:
                            return True
                        print(
                            np.allclose(
                                x, y, rtol=rel_tol, atol=abs_tol, equal_nan=True
                            )
                        )
                        print(x[:5], y[:5])
                        return np.allclose(
                            x, y, rtol=rel_tol, atol=abs_tol, equal_nan=True
                        )

                    diff[key] = DeepDiff(
                        solution_data.__dict__["cell_data"],
                        reference_data.__dict__["cell_data"],
                        iterable_compare_func=custom_compare,
                        # significant_digits=2,
                        number_format_notation="e",
                        ignore_order=True,
                        ignore_numeric_type_changes=True,
                    )

            if files_exist:
                for key in final_solution_filename.keys():
                    if diff[key] != {}:
                        failure.append(f"diff failed for {key}")
                        failure.append(diff[key])
                        if args.verbose:
                            print(f"Diff for {key}:")
                            print(diff[key])

            if failure == []:
                passed.append(combination)
                print(f"Testing: {combination} passed")
            else:
                not_passed.append(combination)
                failure_overview[combination] = failure
                print(f"Testing: {combination} failed")

            if files_exist and formulation == "ncp-min":
                # Compare the solver statistics in terms of number of iterations
                # - NOTE this comparison only makes sense when using the same method!
                solver_statistics = json.loads(solver_statistics_filename.read_text())
                reference_statistics = json.loads(
                    reference_statistics_filename.read_text()
                )
                for time_index in ["1", "2", "3"]:
                    if not (
                        solver_statistics[time_index]["status"]
                        == reference_statistics[time_index]["status"]
                    ):
                        performance_failure.append(
                            f"""Solver status mismatch at time index {time_index}"""
                            f""" ({solver_statistics[time_index]["status"]} vs {reference_statistics[time_index]["status"]})"""
                        )
                    if not (
                        solver_statistics[time_index]["num_iteration"]
                        == reference_statistics[time_index]["num_iteration"]
                    ):
                        performance_failure.append(
                            f"""Solver iterations mismatch at time index {time_index}"""
                            f""" ({solver_statistics[time_index]["num_iteration"]} vs {reference_statistics[time_index]["num_iteration"]})"""
                        )
                    if not (
                        solver_statistics[time_index]["residual_norms"]
                        == reference_statistics[time_index]["residual_norms"]
                    ):
                        performance_failure.append(
                            f"""Solver residual norms mismatch at time index {time_index}"""
                            f""" ({solver_statistics[time_index]["residual_norms"]} vs {reference_statistics[time_index]["residual_norms"]})"""
                        )
            if formulation == "ncp-min":
                if performance_failure == []:
                    performance_passed.append(combination)
                    print(f"Performance testing: {combination} passed")
                else:
                    performance_not_passed.append(combination)
                    performance_failure_overview[combination] = performance_failure
                    print(f"Performance testing: {combination} failed")

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
    f"test_results_3d.txt",
    "a",
) as f:
    f.write("--------------------------------------------------------\n")
    f.write(f"Test run on {datetime.datetime.now()}\n")
    f.write(f"Latest commit message: {latest_commit_message}\n")
    f.write(f"Passed {len(passed)}:\n")
    for item in passed:
        f.write(f"{item}\n")
    f.write(f"\nNot Passed {len(not_passed)}:\n")
    for item in not_passed:
        f.write(f"{item}\n")
    f.write("\nFailure Overview:\n")
    for key, value in failure_overview.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")
    f.write(f"Performance Passed {len(performance_passed)}:\n")
    for item in performance_passed:
        f.write(f"{item}\n")
    f.write(f"\nPerformance Not Passed {len(performance_not_passed)}:\n")
    for item in performance_not_passed:
        f.write(f"{item}\n")
    f.write("\nPerformance Failure Overview:\n")
    for key, value in performance_failure_overview.items():
        f.write(f"{key}: {value}\n")
    f.write("--------------------------------------------------------\n")
