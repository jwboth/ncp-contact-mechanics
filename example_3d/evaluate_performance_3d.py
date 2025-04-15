from pathlib import Path
import json
from main import generate_case_name
import argparse
from icecream import ic
import subprocess
import datetime

argparser = argparse.ArgumentParser(description="Run single fracture test cases.")
argparser.add_argument(
    "-verbose",
    action="store_true",
    help="Print detailed information about the test results.",
)
argparser.add_argument(
    "--num-fractures",
    type=int,
    default=6,
    help="Number of fractures (1-6 [default]).",
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
num_fractures = args.num_fractures
performance = {}

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

            # Read solver statitics
            with open(solver_statistics_filename, "r") as f:
                solver_statistics = json.load(f)

            # Extract the relevant statistics
            num_iterations = 3 * ["nan"]
            for i, time_index in enumerate([1, 2, 3]):
                try:
                    num_iterations[i] = solver_statistics[str(time_index)][
                        "num_iteration"
                    ]
                except KeyError:
                    ...
            performance[combination] = num_iterations


# Print the performance results
ic(performance)


# Fetch latest git log message
def get_latest_git_commit_message():
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%H %B"], capture_output=True, text=True
    )
    return result.stdout.strip()


latest_commit_message = get_latest_git_commit_message()

# Report the results in txt file annotated by the date and time - append if the file exists
with open(
    f"performance_results_3d_{args.num_fractures}.txt",
    "a",
) as f:
    f.write("--------------------------------------------------------\n")
    f.write(f"Test run on {datetime.datetime.now()}\n")
    f.write(f"Latest commit message: {latest_commit_message}\n")
    f.write(f"Performance {len(performance)}:\n")
    for item in performance:
        f.write(f"{item}\n")
    f.write("--------------------------------------------------------\n")
