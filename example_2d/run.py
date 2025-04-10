"""Example 2.

* Run all studies with all methods and seeds.
* Investigate the impact of the complexity of the fracture network (study is proxy.)


"""

import sys
import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run single fracture test cases.")
parser.add_argument("-o", "--option", type=str, default="all", help="Option to run.")
args = parser.parse_args()

# ! ---- Fixed parameters ----

methods = [
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
all_studies = np.arange(2, 4, dtype=int)
all_seeds = np.arange(0, 3, dtype=int)

# Keep constant for all runs
unitary_units = True
dil = 0.1
keep_all_intersections = [(True, 0.0)]


# ! ---- Options ----

studies = all_studies
seeds = all_seeds
methods = methods
linear_solvers = ["scipy_sparse-pypardiso"]
mesh_sizes = [10]
intersections_options = keep_all_intersections
num_time_steps = 5
num_iter = 200
cnums = [1.0]
tols = [1e-10]
output = "visualization"

for study in studies:
    for seed in seeds:
        for ad_mode, mode, aa, regularization in methods:
            for linear_solver in linear_solvers:
                for (
                    no_intersections,
                    no_intersections_angle_cutoff,
                ) in intersections_options:
                    for mesh_size in mesh_sizes:
                        for cnum in cnums:
                            for tol in tols:
                                instructions = [
                                    sys.executable,
                                    "main.py",
                                    "--study",
                                    str(study),
                                    "--ad-mode",
                                    ad_mode,
                                    "--mode",
                                    mode,
                                    "--linear-solver",
                                    linear_solver,
                                    "--aa",
                                    str(aa),
                                    "--regularization",
                                    regularization,
                                    "--tol",
                                    str(tol),
                                    "--cn",
                                    str(cnum),
                                    "--ct",
                                    str(cnum),
                                    "--unitary_units",
                                    str(unitary_units),
                                    "--seed",
                                    str(seed),
                                    "--no_intersections",
                                    str(no_intersections),
                                    "--no_intersections_angle_cutoff",
                                    str(no_intersections_angle_cutoff),
                                    "--mesh_size",
                                    str(mesh_size),
                                    "--output",
                                    output,
                                    "--num_time_steps",
                                    str(num_time_steps),
                                    "--num_iter",
                                    str(num_iter),
                                ]
                                subprocess.run(instructions)
