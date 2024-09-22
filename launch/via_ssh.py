import argparse
from datetime import datetime
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

    msg = "File contents:\n" + textwrap.indent(text, "> ")
    logging.info(msg)

    CMD = ["ssh", host, "mktemp", f"{prefix}.XXXXXXXX", "-p", "/tmp"]
    logging.info(shlex.join(CMD))
    remote_file = subprocess.check_output(CMD).decode().rstrip()

    CMD = ["scp", host_file, f"{host}:{remote_file}"]
    logging.info(shlex.join(CMD))
    subprocess.run(CMD, stdout=DEVNULL).check_returncode()

    Path(host_file).unlink()

    return remote_file


def remote_exec(host: str, script: str):
    msg = "Script:\n" + textwrap.indent(script, "> ")
    logging.info(msg)

    CMD = ["ssh", host, "bash", "-i", "-s"]
    logging.info(shlex.join(CMD))

    proc = subprocess.Popen(
        ["ssh", host, "bash", "-i", "-s"],
        stdin=subprocess.PIPE,
    )
    proc.communicate((script + "\n").encode("utf-8"))
    if proc.returncode:
        raise RuntimeError("Process retuned non-0 return code.")


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

    launch_p = sp.add_parser(
        "launch",
        help="Distribute a list of commands over a number of hosts.",
    )
    launch_p.add_argument(
        "--hosts",
        nargs="+",
        type=str,
        help="A list of ssh hosts to deploy the commands to",
    )
    launch_p.add_argument(
        "--git-ref",
        help="Git commit ref (e.g. branch, or SHA). If not provided, use current branch.",
    )
    launch_p.add_argument(
        "-s",
        "--session",
        type=str,
        required=True,
        help="tmux session name",
    )
    launch_p.add_argument(
        "-j",
        "--tasks-per-host",
        default=1,
        type=int,
    )
    launch_p.add_argument(
        "--unsafe",
        action="store_true",
        help="Allow launch even if the repo is dirty",
    )
    launch_p.add_argument(
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
        "-i",
        "--info-yml",
        type=Path,
        required=True,
        help="Path to info.yml file for the run",
    )

    status_p = sp.add_parser(
        "status",
        help="Get status of a run.",
    )
    status_p.add_argument(
        "-i",
        "--info-yml",
        type=Path,
        required=True,
        help="Path to info.yml file for the run",
    )

    gather_p = sp.add_parser(
        "gather",
        help="Gather results from the tests.",
    )
    gather_p.add_argument(
        "-i",
        "--info-yml",
        type=Path,
        required=True,
        help="Path to info.yml file for the run.",
    )
    gather_p.add_argument(
        "--run-dir-key",
        default="run.dir",
        help="Location of the run exp directory in the config object.",
    )
    gather_p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory.",
    )

    args = p.parse_args()

    def _check_availability(hosts: list[str]):
        for host in hosts:
            CMD = ["ssh", host, "true"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD, stdout=DEVNULL).check_returncode()

    def sync_hosts(hosts=None):
        if hosts is None:
            hosts = args.hosts

        _check_availability(hosts)

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
            remote_exec(host, sync_host)

    def fail_if_dirty():
        repo = git.Repo(
            path=Path(__file__).parent,
            search_parent_directories=True,
        )
        if repo.is_dirty():
            logging.error("Repo CANNOT be dirty when running grid launches!")
            exit(1)

    def grid_launch():
        if not args.unsafe:
            fail_if_dirty()

        hosts = args.hosts
        all_cmds = [line.strip() for line in args.commands]

        if len(all_cmds) < len(hosts):
            hosts = hosts[: len(all_cmds)]
            cmd_map = {host: [cmd] for host, cmd in zip(hosts, all_cmds)}
        else:
            div_ = len(all_cmds) // len(hosts)
            rem_ = len(all_cmds) % len(hosts)
            pivots = [0]
            for idx in range(len(hosts)):
                pivots.append(pivots[-1] + div_ + (1 if idx < rem_ else 0))
            cmd_map = {
                host: all_cmds[start:end]
                for host, start, end in zip(hosts, pivots, pivots[1:])
            }

        sync_hosts(hosts)

        for host, cmds in cmd_map.items():
            cmd_path = remote_dump(host, "\n".join(cmds), "cmd")

            tmux_script = f"""
            cd "{args.repo_dir}"
            source $(poetry env info --path)/bin/activate
            parallel -j{args.tasks_per_host} -a "{cmd_path}"
            """
            tmux_script = inspect.cleandoc(tmux_script)
            tmux_path = remote_dump(host, tmux_script, "tmux")

            remote_exec(
                host,
                f'tmux new-session -s "{args.session}" -d "source {tmux_path}"',
            )

        dt = datetime.now()
        info_path = Path(f"runs/{args.session}.{dt:%Y-%m-%d_%H-%M-%S}.yml")

        info = {
            "session": args.session,
            "hosts": {host: cmds for host, cmds in cmd_map.items()},
        }
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, "w") as f:
            yaml.dump(info, f)

        logging.info(f"Run info saved to: {info_path}")

    def kill():
        with open(args.info_yml) as f:
            info = yaml.load(f)

        for host in info["hosts"]:
            CMD = ["ssh", host, "tmux", "send-key", "-t", info["session"], "C-c"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD)

    def status():
        with open(args.info_yml) as f:
            info = yaml.load(f)

        for host in info["hosts"]:
            CMD = ["ssh", host, "tmux", "ls"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD)

            CMD = ["ssh", host, "nvidia-smi"]
            logging.info(shlex.join(CMD))
            subprocess.run(CMD).check_returncode()

    def gather():
        with open(args.info_yml) as f:
            info = yaml.load(f)

        for host, cmds in info["hosts"].items():
            for cmd in cmds:
                argv = shlex.split(cmd)
                CMD = [*argv, "--dump-config"]
                logging.info(shlex.join(CMD))
                cfg_bytes = subprocess.check_output(CMD)

                cfg = yaml.load(io.BytesIO(cfg_bytes))

                cur = cfg
                for key in args.run_dir_key.split("."):
                    cur = cur[key]
                run_dir = cur

                args.output_dir.mkdir(parents=True, exist_ok=True)

                dst_dir = args.output_dir / Path(run_dir).name
                if dst_dir.exists():
                    idx = 0
                    while True:
                        old_dir = dst_dir.with_name(dst_dir.name + f".old{idx:03d}")
                        if not old_dir.exists():
                            break

                    dst_dir.rename(old_dir)

                CMD = [
                    "rsync",
                    "-ruz",
                    "--delete",
                    "--info=progress2",
                    f"{host}:~/self/rsrch/{run_dir}/",
                    str(dst_dir),
                ]
                logging.info(shlex.join(CMD))
                subprocess.run(CMD).check_returncode()

    if args.mode == "sync":
        sync_hosts()
    elif args.mode == "launch":
        grid_launch()
    elif args.mode == "kill":
        kill()
    elif args.mode == "status":
        status()
    elif args.mode == "gather":
        gather()


if __name__ == "__main__":
    main()
