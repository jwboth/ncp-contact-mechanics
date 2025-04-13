import subprocess
import sys
from pathlib import Path
from deepdiff import DeepDiff
import json
from main import generate_case_name
import meshio

# Test different formulations
formulations = [
    ("picard", "rr-nonlinear", 0, "none"),
    ("picard", "rr-linear", 0, "none"),
    ("picard", "ncp-min", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-full", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-min-scaled", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-scaled", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-full-scaled", 0, "origin_and_stick_slip_transition"),
    #("newton", "rr-nonlinear", 0, "none"),
    #("newton", "rr-linear", 0, "none"),
    #("newton", "ncp-min", 0, "origin_and_stick_slip_transition"),
    #("newton", "ncp-fb-full", 0, "origin_and_stick_slip_transition"),
    #("newton", "ncp-min-scaled", 0, "origin_and_stick_slip_transition"),
    #("newton", "ncp-fb-scaled", 0, "origin_and_stick_slip_transition"),
    #("newton", "ncp-fb-full-scaled", 0, "origin_and_stick_slip_transition"),
]
study = 2
seed = 4
passed = []
not_passed = formulations.copy()
not_passed_reason = {}

for formulation in formulations:
    # Run the simulation with the specified formulation
    print(f"Testing formulation: {formulation}")
    ad_mode, mode, aa, regularization = formulation

    # Fetch the solver statistics
    folder = Path("visualization") / generate_case_name(
        study=study,
        seed=seed,
        mesh_size=10,
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
    solver_statistics_filename = folder / "solver_statistics.json"
    final_solution_filename = {
        "data_1": folder / "data_1_000003.vtu",
        "data_2": folder / "data_2_000003.vtu",
        "mortar_1": folder / "data_mortar_1_000003.vtu",
    }

    # Fetch references
    reference_statistics_filename = Path("reference/solver_statistics.json")
    reference_solution_filename = {
        "data_1": Path("reference/data_1_000003.vtu"),
        "data_2": Path("reference/data_2_000003.vtu"),
        "mortar_1": Path("reference/data_mortar_1_000003.vtu"),
    }

    # Compare the final solution files
    diff = {}
    try:
        for key in final_solution_filename.keys():
            # Compare the files
            diff[key] = DeepDiff(
                meshio.read(final_solution_filename[key]).__dict__,
                meshio.read(reference_solution_filename[key]).__dict__,
                significant_digits = 3,
                number_format_notation = "e",
                ignore_order=True,
                ignore_numeric_type_changes=True,
            )
    except:
        ...
    try:
        # Compare the solver statistics in terms of number of iterations
        # - NOTE this comparison only makes sense when using the same method!
        if formulation == "ncp-min-scaled":
            solver_statistics = json.loads(solver_statistics_filename.read_text())
            reference_statistics = json.loads(reference_statistics_filename.read_text())
            for time_index in ["1", "2", "3"]:
                assert (
                    solver_statistics[time_index]["status"]
                    == reference_statistics[time_index]["status"]
                ), f"Solver status mismatch at time index {time_index}"
                assert (
                    solver_statistics[time_index]["num_iteration"]
                    == reference_statistics[time_index]["num_iteration"]
                ), f"Solver iterations mismatch at time index {time_index}"
                assert (
                    solver_statistics[time_index]["residual_norms"]
                    == reference_statistics[time_index]["residual_norms"]
                ), f"Solver residual norms mismatch at time index {time_index}"

        # Compare the final solution files
        if diff == {}:
            assert False, f"Files missing for {formulation}"

        for key in final_solution_filename.keys():
            # Check if there are any differences
            assert not diff[key], (
                f"Files {final_solution_filename[key]} and {reference_solution_filename[key]} differ: {diff}"
            )

        print("All tests passed for formulation:", formulation)
        passed.append(formulation)
        not_passed.remove(formulation)
    except AssertionError as e:
        print(f"Test failed for formulation {formulation}: {e}")
        not_passed_reason[formulation] = e
    except FileNotFoundError as e:
        print(f"File not found for formulation {formulation}: {e}")
        not_passed_reason[formulation] = e
    except Exception as e:
        print(f"An unexpected error occurred for formulation {formulation}: {e}")
        not_passed_reason[formulation] = e

# Print the results
print("Passed formulations:", passed)
print("Not passed formulations:", not_passed)
print("Not passed reasons:", not_passed_reason)
