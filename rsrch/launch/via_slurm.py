import argparse
import shlex
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="A helper for deploying grid launches via SLURM. Pass extra arguments to sbatch by after -- (e.g. via_slurm.py -- -J <job_name>)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Whether to only print the command to execute.",
    )
    p.add_argument(
        "--commands",
        default="commands.txt",
        help="Path to a file with commands for the grid launch. Each line should contain a command to execute. Default: ./commands.txt",
    )

    args, extra = p.parse_known_args()

    if len(extra) > 0:
        index = next((idx for idx, s in enumerate(extra) if s == "--"), len(extra))
        if index != 0:
            raise ValueError(f"Unknown arguments: {shlex.join(extra[:index])}")
        extra = extra[1:]

    with open(args.commands, "r") as f:
        num_cmd = sum(1 for _ in f)

    grid_slurm = Path(__file__).parent / "grid.slurm"

    CMD = [
        "sbatch",
        f"--array=1-{num_cmd}",
        *extra,
        str(grid_slurm),
    ]
    if args.dry_run:
        print(shlex.join(CMD))
    else:
        subprocess.run(CMD)


if __name__ == "__main__":
    main()
