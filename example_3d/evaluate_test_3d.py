from pathlib import Path
from deepdiff import DeepDiff
import json
from main import generate_case_name
import meshio
import argparse
from icecream import ic
import datetime

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
    "ncp-min-scaled",
    "ncp-fb-full",
    "ncp-fb-full-scaled",
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
                Path("reference") / folder.parent / "solver_statistics.json"
            )
            reference_solution_filename = {
                "data_2": Path("reference") / folder.parent / "data_2_000003.vtu",
                "data_3": Path("reference") / folder.parent / "data_3_000003.vtu",
                "mortar_2": Path("reference")
                / folder.parent
                / "data_mortar_2_000003.vtu",
            }

            # Fill in with the missing data_1 and mortar_1 files if the exist
            if (Path("visualization") / folder / "data_1_000000.vtu").exists():
                final_solution_filename["data_1"] = (
                    Path("visualization") / folder / "data_1_000003.vtu"
                )
                reference_solution_filename["data_1"] = (
                    Path("reference") / folder.parent / "data_1_000003.vtu"
                )
            if (Path("visualization") / folder / "data_mortar_1_000000.vtu").exists():
                final_solution_filename["mortar_1"] = (
                    Path("visualization") / folder / "data_mortar_1_000003.vtu"
                )
                reference_solution_filename["mortar_1"] = (
                    Path("reference") / folder.parent / "data_mortar_1_000003.vtu"
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

            if failure == []:
                passed.append(formulation)
                print(f"Testing formulation: {formulation} passed")
            else:
                not_passed.append(formulation)
                failure_overview[formulation] = failure
                print(f"Testing formulation: {formulation} failed")

            if files_exist and formulation == "ncp-min-scaled":
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
            if formulation == "ncp-min-scaled":
                if performance_failure == []:
                    performance_passed.append(formulation)
                    print(f"Performance testing formulation: {formulation} passed")
                else:
                    performance_not_passed.append(formulation)
                    performance_failure_overview[formulation] = failure
                    print(f"Performance testing formulation: {formulation} failed")

# Print the results
ic(passed)
ic(not_passed)
ic(failure_overview)
ic(performance_passed)
ic(performance_not_passed)
ic(performance_failure_overview)

# Report the results in txt file annotated by the date and time - append if the file exists
with open(
    f"test_results_3d.txt",
    "a",
) as f:
    f.write("--------------------------------------------------------\n")
    f.write(f"Test run on {datetime.datetime.now()}\n")
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
