import subprocess
import sys
from pathlib import Path
from deepdiff import DeepDiff
import json

# Test different formulations
for formulation in ["ncp-min-scaled", "ncp-fb-scaled", "rr-linear", "rr-nonlinear"]:
    # Run the simulation with the specified formulation
    print(f"Testing formulation: {formulation}")
    subprocess.run(
        [
            sys.executable,
            "main.py",
            "--formulation",
            "ncp-min-scaled",
            "--mass-unit",
            "1",
        ]
    )

    # Fetch the solver statistics
    folder = Path("visualization/simple_bedretto_6/ncp-min-scaled_picard_scipy_sparse")
    solver_statistics_filename = folder / "solver_statistics.json"
    final_solution_filename = {
        "data_1": folder / "data_1_000003.vtu",
        "data_2": folder / "data_2_000003.vtu",
        "data_3": folder / "data_3_000003.vtu",
        "mortar_1": folder / "data_mortar_1_000003.vtu",
        "mortar_2": folder / "data_mortar_2_000003.vtu",
    }

    # Fetch references
    reference_statistics_filename = Path("reference/solver_statistics.json")
    reference_solution_filename = {
        "data_1": Path("reference/data_1_000003.vtu"),
        "data_2": Path("reference/data_2_000003.vtu"),
        "data_3": Path("reference/data_3_000003.vtu"),
        "mortar_1": Path("reference/data_mortar_1_000003.vtu"),
        "mortar_2": Path("reference/data_mortar_2_000003.vtu"),
    }

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
    for key in final_solution_filename.keys():
        # Compare the files
        diff = DeepDiff(
            final_solution_filename[key].read_text(),
            reference_solution_filename[key].read_text(),
            ignore_order=True,
        )
        # Check if there are any differences
        assert not diff, (
            f"Files {final_solution_filename[key]} and {reference_solution_filename[key]} differ: {diff}"
        )

print("All tests passed!")
