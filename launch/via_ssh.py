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

import git
import jinja2
import paramiko


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)-13s: %(levelname)-8s %(message)s",
    )

    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode")

    sync_p = sp.add_parser(
        "sync",
        help="Sync hosts, ensuring that the project repo is set up and up to date.",
    )
    sync_p.add_argument(
        "--hosts",
        nargs="+",
        type=str,
        help="A list of ssh hosts to deploy the commands to",
    )
    sync_p.add_argument(
        "--git-ref",
        help="Git commit ref (e.g. branch, or SHA). If not provided, use current branch.",
    )

    grid_p = sp.add_parser(
        "grid",
        help="Distribute a list of commands over a number of hosts.",
    )
    grid_p.add_argument(
        "--hosts",
        nargs="+",
        type=str,
        help="A list of ssh hosts to deploy the commands to",
    )
    grid_p.add_argument(
        "--git-ref",
        help="Git commit ref (e.g. branch, or SHA). If not provided, use current branch.",
    )
    grid_p.add_argument(
        "-s",
        "--session",
        type=str,
        required=True,
        help="tmux session name",
    )
    grid_p.add_argument(
        "-j",
        "--tasks-per-host",
        default=1,
        type=int,
    )
    grid_p.add_argument(
        "--unsafe",
        action="store_true",
        help="Allow launch even if the repo is dirty",
    )
    grid_p.add_argument(
        "commands",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="A file with commands. If '-' or not provided, read the commands from stdin.",
    )

    args = p.parse_args()

    # For some reason, trying to read args.commands after sync_hosts call
    # results in [], as though the call cleared stdin.
    if args.mode == "grid":
        all_cmds = [line.strip() for line in args.commands]

    hosts = args.hosts
    for host in hosts:
        CMD = ["ssh", host]
        logging.info(shlex.join(CMD))
        subprocess.run(CMD, stdout=DEVNULL).check_returncode()

    def sync_hosts():
        repo = git.Repo(
            path=Path(__file__).parent,
            search_parent_directories=True,
        )

        git_repo_url = repo.remote("origin").url
        git_user_name = repo.config_reader().get_value("user", "name")
        git_user_email = repo.config_reader().get_value("user", "email")
        git_ref = args.git_ref or repo.head.ref.name

        env = jinja2.Environment()
        loader = jinja2.FileSystemLoader(Path(__file__).parent)
        sync_host_tmpl = loader.load(env, "sync_host.j2")

        sync_host = sync_host_tmpl.render(
            git_repo_url=git_repo_url,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
            git_ref=git_ref,
        )

        sync_host_path = tempfile.mktemp(prefix="sync_host.")
        with open(sync_host_path, "w") as f:
            f.write(sync_host)

        for host in hosts:
            CMD = ["ssh", host, "mktemp", "sync_host.XXXXXXXX", "-p", "/tmp"]
            logging.info(shlex.join(CMD))
            remote_sync = subprocess.check_output(CMD).decode().rstrip()

            CMD = ["scp", sync_host_path, f"{host}:{remote_sync}"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD, stdout=DEVNULL).check_returncode()

            CMD = ["ssh", host, "bash", "-i", remote_sync]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD).check_returncode()

            CMD = ["ssh", host, "rm", remote_sync]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD).check_returncode()

    def fail_if_dirty():
        repo = git.Repo(
            path=Path(__file__).parent,
            search_parent_directories=True,
        )
        if repo.is_dirty():
            logging.error("Repo CANNOT be dirty when running grid launches!")
            exit(1)

    def grid_launch():
        nonlocal all_cmds
        pivots = [int((i * len(all_cmds)) / len(hosts)) for i in range(len(hosts) + 1)]
        all_cmds = [all_cmds[start:end] for start, end in zip(pivots, pivots[1:])]

        for host, cmds in zip(hosts, all_cmds):
            CMD = ["ssh", host, "mktemp", "cmd.XXXXXXXX", "-p", "/tmp"]
            logging.info(shlex.join(CMD))
            remote_cmd = subprocess.check_output(CMD).decode().rstrip()

            host_cmd = tempfile.mktemp(prefix="cmd.")
            with open(host_cmd, "w") as f:
                f.writelines(cmd + "\n" for cmd in cmds)

            CMD = ["scp", host_cmd, f"{host}:{remote_cmd}"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD, stdout=DEVNULL).check_returncode()

            CMD = [
                "ssh",
                host,
                "tmux",
                "new-session",
                "-s",
                args.session,
                "-d",
                f"parallel -j{args.tasks_per_host} -a {remote_cmd}",
            ]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD).check_returncode()

    if args.mode == "sync":
        sync_hosts()
    elif args.mode == "grid":
        if not args.unsafe:
            fail_if_dirty()
        sync_hosts()
        grid_launch()


if __name__ == "__main__":
    main()
