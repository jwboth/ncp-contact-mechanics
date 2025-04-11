import sys
import subprocess

# import argparse
#
# parser = argparse.ArgumentParser(description="Run single fracture test cases.")
# parser.add_argument("-o", "--option", type=str, default="all", help="Option to run.")
# args = parser.parse_args()

# ! ---- Options ----

formulations = [
    "rr-nonlinear",
    "rr-linear",
    "ncp-min",
]

for formulation in formulations:
    instructions = [
        sys.executable,
        "main.py",
        "--formulation",
        formulation,
    ]
    subprocess.run(instructions)
