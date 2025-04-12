from pathlib import Path
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run list of instructions")
parser.add_argument(
    "--path", type=str, required=True, help="path to the file with instructions"
)
args = parser.parse_args()

path = args.path
with open(path, "r") as f:
    instructions = f.readlines()
    log = []
    for instruction in instructions:
        # Run instruction using subprocess
        print(instruction.strip())
        subprocess.run(instruction.strip(), shell=True)
        log.append(instruction.strip())
        # Write to log file
        Path("cache/log").mkdir(parents=True, exist_ok=True)
        with open(f"cache/log/{Path(path).stem}.txt", "a") as log_file:
            log_file.write(instruction.strip() + "\n")
