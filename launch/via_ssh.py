import argparse
import inspect
import io
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from subprocess import DEVNULL, PIPE

import git
import jinja2
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)


def remote_dump(host: str, text: str, prefix: str):
    """Given an SSH host, text and a prefix, dump contents of the stream to a temp file on the host, and return the path."""

    host_file = tempfile.mktemp()
    with open(host_file, "w") as f:
        f.write(text)

    msg = "File contents:\n" + textwrap.indent(text, " " * 4)
    logging.info(msg)

    CMD = ["ssh", host, "mktemp", f"{prefix}.XXXXXXXX", "-p", "/tmp"]
    logging.info(shlex.join(CMD))
    remote_file = subprocess.check_output(CMD).decode().rstrip()

    CMD = ["scp", host_file, f"{host}:{remote_file}"]
    logging.info(shlex.join(CMD))
    subprocess.run(CMD, stdout=DEVNULL).check_returncode()

    Path(host_file).unlink()

    return remote_file


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)-13s: %(levelname)-8s %(message)s",
    )

    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode")

    p.add_argument(
        "--repo-dir",
        type=str,
        default="$HOME/self/rsrch",
        help="Repo dir to use. NOTE: Do not use ~ for $HOME.",
    )

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
        "--dump-info",
        type=Path,
        help="(Optional) Path for the yaml file with info about the launch",
    )
    grid_p.add_argument(
        "commands",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="A file with commands. If '-' or not provided, read the commands from stdin.",
    )

    kill_p = sp.add_parser(
        "kill",
        help="Kill running grid launch.",
    )
    kill_p.add_argument(
        "--hosts",
        nargs="+",
        type=str,
        help="A list of ssh hosts to deploy the commands to",
    )
    kill_p.add_argument(
        "-s",
        "--session",
        type=str,
        required=True,
        help="tmux session name",
    )

    args = p.parse_args()

    # For some reason, trying to read args.commands after sync_hosts call
    # results in [], as though the call cleared stdin.
    if args.mode == "grid":
        all_cmds = [line.strip() for line in args.commands]

    hosts = args.hosts
    for host in hosts:
        CMD = ["ssh", host, "true"]
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
        sync_host_tmpl = loader.load(env, "sync_host.sh.j2")

        sync_host = sync_host_tmpl.render(
            git_repo_url=git_repo_url,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
            git_ref=git_ref,
            repo_dir=args.repo_dir,
        )

        for host in hosts:
            script_path = remote_dump(host, sync_host, "sync_host")

            CMD = ["ssh", host, "bash", "-i", script_path]
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
            cmd_path = remote_dump(host, "\n".join(cmds), "cmd")

            tmux_script = f"""
            cd "{args.repo_dir}"
            source $(poetry env info --path)/bin/activate
            parallel -j{args.tasks_per_host} -a "{cmd_path}"
            """
            tmux_script = inspect.cleandoc(tmux_script)
            tmux_path = remote_dump(host, tmux_script, "tmux")

            exec_script = (
                f'tmux new-session -s "{args.session}" -d "source {tmux_path}"'
            )
            exec_path = remote_dump(host, exec_script, "exec")

            CMD = ["ssh", host, "bash", "-i", exec_path]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD).check_returncode()

        if args.dump_info is not None:
            info = {
                "session": args.session,
                "hosts": {host: "\n".join(cmds) for host, cmds in zip(hosts, all_cmds)},
            }
            with open(args.dump_info, "w") as f:
                yaml.dump(info, f)

    def kill():
        for host in hosts:
            CMD = ["ssh", host, "tmux", "send-key", "-t", args.session, "C-c"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD).check_returncode()

    if args.mode == "sync":
        sync_hosts()
    elif args.mode == "grid":
        if not args.unsafe:
            fail_if_dirty()
        sync_hosts()
        grid_launch()
    elif args.mode == "kill":
        kill()


if __name__ == "__main__":
    main()
