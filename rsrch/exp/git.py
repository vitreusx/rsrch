from pathlib import Path

import git
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)


def create_exp_commit(run: str) -> str:
    """Create a commit for an experiment in a separate branch named
    `exp/{branch}`, where {branch} is the current branch name."""

    repo = git.Repo(Path(__file__).parent, search_parent_directories=True)

    # Get current branch name
    cur_ref: str = repo.git.rev_parse("--abbrev-ref", "HEAD")

    if cur_ref.startswith("exp/"):
        # If the process is killed during the checkout process, we'll get stuck
        # in an internal exp/* branch, and with .git/index.lock.
        (Path(repo.git_dir) / "index.lock").unlink(missing_ok=True)
        correct_ref = cur_ref.removeprefix("exp/")
        repo.git.symbolic_ref("HEAD", f"refs/heads/{correct_ref}")
        repo.git.reset()
        cur_ref = correct_ref

    exp_ref = f"exp/{cur_ref}"

    try:
        # Create auto/{branch} if necessary
        if not any(ref.name == exp_ref for ref in repo.refs):
            repo.git.checkout("-b", exp_ref)

        # Switch HEAD to auto/{branch}, including index but without modifying
        # the working tree
        repo.git.symbolic_ref("HEAD", f"refs/heads/{exp_ref}")
        repo.git.add("--all")
        commit_msg = f"Exp commit for {run}"
        repo.git.commit("--allow-empty", "-m", commit_msg)
        commit_sha = repo.head.object.hexsha
    finally:
        # Switch back to the original state
        repo.git.symbolic_ref("HEAD", f"refs/heads/{cur_ref}")
        repo.git.reset()

    return commit_sha


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--reset",
        action="store_true",
        help="Fix aborted switch to exp/* branch.",
    )
    p.add_argument(
        "--restore",
        help="Restore files for a run. Either run dir or commit SHA.",
    )

    args = p.parse_args()

    repo = git.Repo(Path(__file__).parent, search_parent_directories=True)

    if args.reset:
        cur_ref: str = repo.git.rev_parse("--abbrev-ref", "HEAD")
        if cur_ref.startswith("exp/"):
            # If the process is killed during the checkout process, we'll get stuck
            # in an internal exp/* branch, and with .git/index.lock.
            (Path(repo.git_dir) / "index.lock").unlink(missing_ok=True)
            correct_ref = cur_ref.removeprefix("exp/")
            repo.git.symbolic_ref("HEAD", f"refs/heads/{correct_ref}")
            repo.git.reset()
    elif args.restore is not None:
        info_path = Path(args.restore) / "info.yml"
        if info_path.exists():
            with open(info_path, "r") as f:
                commit_sha = yaml.load(f)["commit_sha"]
        else:
            commit_sha = args.restore

        repo.git.restore("-s", commit_sha, ".")


if __name__ == "__main__":
    main()
