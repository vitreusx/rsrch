import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from subprocess import DEVNULL


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--session",
        type=str,
        required=True,
        help="tmux session name",
    )
    p.add_argument(
        "-j",
        "--tasks-per-host",
        default=1,
        type=int,
    )
    p.add_argument(
        "--hosts",
        nargs="+",
        type=str,
        help="A list of ssh hosts to deploy the commands to",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)-13s: %(levelname)-8s %(message)s",
    )

    args = p.parse_args()

    hosts = args.hosts

    cmds = [line.strip() for line in sys.stdin]

    pivots = [int((i * len(cmds)) / len(hosts)) for i in range(len(hosts) + 1)]
    cmds = [cmds[start:end] for start, end in zip(pivots, pivots[1:])]

    sync_host_path = str(Path(__file__).parent / "sync_host")

    for idx, host in enumerate(hosts):
        CMD = ["ssh", host, "mktemp", "sync_host.XXXXXXXX", "-p", "/tmp"]
        logging.info(shlex.join(CMD))
        remote_sync = subprocess.check_output(CMD).decode().rstrip()

        CMD = ["ssh", host, "mktemp", "cmd.XXXXXXXX", "-p", "/tmp"]
        logging.info(shlex.join(CMD))
        remote_cmd = subprocess.check_output(CMD).decode().rstrip()

        CMD = ["scp", sync_host_path, f"{host}:{remote_sync}"]
        logging.info(shlex.join(CMD))
        subprocess.run(CMD)

        host_cmd = tempfile.mktemp(prefix="cmd.")
        with open(host_cmd, "w") as f:
            f.writelines(cmd + "\n" for cmd in cmds[idx])

        CMD = ["scp", host_cmd, f"{host}:{remote_cmd}"]
        logging.info(shlex.join(CMD))
        subprocess.run(CMD)

        CMD = ["ssh", host, "bash", "-i", remote_sync]
        logging.info(shlex.join(CMD))
        subprocess.run(CMD)

        CMD = [
            "ssh",
            host,
            "tmux",
            "new-session",
            "-A",
            "-s",
            args.session,
            "-d",
            f"parallel -j{args.tasks_per_host} -a {remote_cmd}",
        ]
        logging.info(shlex.join(CMD))
        subprocess.run(CMD)


if __name__ == "__main__":
    main()
