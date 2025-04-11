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
    "ncp-min-scaled",
    "ncp-fb-full",
    "ncp-fb-full-scaled",
]

mass_units = [1e0, 1e6, 1e10]

for formulation in formulations:
    for mass_unit in mass_units:
        instructions = [
            sys.executable,
            "main.py",
            "--formulation",
            formulation,
            "--mass-unit",
            str(mass_unit),
        ]
        subprocess.run(instructions)
