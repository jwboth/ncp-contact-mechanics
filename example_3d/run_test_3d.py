import subprocess
import shutil
import sys
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser(description="Run single fracture test cases.")
parser.add_argument("-dry", "--dry", action="store_true", help="Only dry run.")
parser.add_argument(
    "-c", "--cache", type=str, default="cache", help="cache for parallel runs."
)
parser.add_argument(
    "-p", "--parallel", action="store_true", help="run in parallel mode."
)
parser.add_argument(
    "-np",
    "--parallel-processors",
    type=int,
    nargs="+",
    default=[-1, -1],
    help="start and end.",
)
parser.add_argument(
    "-p-list",
    "--parallel-processors-list",
    type=int,
    nargs="+",
    default=[-1, -1],
    help="selected processor list",
)

args = parser.parse_args()

# Test different formulations
formulations = [
    "rr-nonlinear",
    "rr-nonlinear-unscaled",
    "rr-linear",
    "rr-linear-unscaled",
    "ncp-min",
    "ncp-min-scaled",
    "ncp-fb-full",
    "ncp-fb-full-scaled",
]
passed = []
not_passed = formulations.copy()
not_passed_reason = {}

pool_instructions = []

for apply_horzizontal_stress in [True, False]:
    for mass_unit in [1, 1e10]:
        for formulation in formulations:
            # Run the simulation with the specified formulation
            print(f"Testing formulation: {formulation}")
            instructions = [
                sys.executable,
                "main.py",
                "--formulation",
                formulation,
                "--mass-unit",
                str(mass_unit),
                "--asci-export",
                "--num-fractures",
                "2",
            ]
            if apply_horzizontal_stress:
                instructions += ["--apply-horizontal-stress"]

            if args.parallel:
                pool_instructions.append(instructions)
            else:
                subprocess.run(instructions)

# Coordinate parallel runs using 'nohup taskset --cpu-list N python instructions (unrolled)'
# Use for N in range(args.parallel_processors[0], args.parallel_processors[1]+1)
# fill each cpu with a task, then move to next cpu after finished
if args.parallel:
    from icecream import ic

    for i, pi in enumerate(pool_instructions):
        print(i, pi)

    # Distribute the tasks among the available processors
    available_cpus = [
        i for i in range(args.parallel_processors[0], args.parallel_processors[1] + 1)
    ]
    available_cpus += args.parallel_processors_list
    available_cpus = list(set(available_cpus))
    available_cpus = [proc for proc in available_cpus if proc > 0]
    num_available_cpus = len(available_cpus)
    split_instructions = {}
    for i, instruction in enumerate(pool_instructions):
        # Determine which cpu to use
        cpu = available_cpus[i % num_available_cpus]
        # Store the instruction in the split_instructions
        if cpu not in split_instructions:
            split_instructions[cpu] = []
        split_instructions[cpu].append(instruction)

    # Stop if dry run
    assert not (args.dry), "Only dry run."

    # Remove cache directory if it exists
    cache_dir = Path(args.cache)
    if cache_dir.exists():
        # Remove cache_dir using shutil
        shutil.rmtree(str(cache_dir))
    # Create cache directory if it does not exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Store the instructions to a file cache/parallel_instruction_processor_N.txt
    for cpu, instructions in split_instructions.items():
        with open(f"{args.cache}/parallel_instruction_processor_{cpu}.txt", "w") as f:
            for instruction in instructions:
                f.write(" ".join(instruction) + "\n")

    pool_instructions = list(cache_dir.glob("*.txt"))
    for i, instruction_file in enumerate(pool_instructions):
        # To mitigate racing conditions, wait 5 seconds for each additional run
        time.sleep(5)

        processor = instruction_file.stem.split("_")[-1]

        # Use nohup taskset to run the instruction file on the specified processor
        subprocess.Popen(
            " ".join(
                [
                    "nohup",
                    "taskset",
                    "--cpu-list",
                    str(int(processor) - 1),
                    sys.executable,
                    "run_instructions.py",
                    "--path",
                    str(instruction_file),
                    ">",
                    f"nohup_{processor}.out",
                    "2>&1",
                    "&",
                ]
            ),
            shell=True,
        )
