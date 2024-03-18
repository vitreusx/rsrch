from pathlib import Path

import git


def create_exp_commit(run: str) -> str:
    """Create a commit for an experiment in a separate branch named
    `exp/{branch}`, where {branch} is the current branch name."""

    repo = git.Repo(Path(__file__).parent, search_parent_directories=True)

    # Get current branch name
    cur_ref = repo.git.rev_parse("--abbrev-ref", "HEAD")
    exp_ref = f"exp/{cur_ref}"

    # Create auto/{branch} if necessary
    if not any(ref.name == exp_ref for ref in repo.refs):
        repo.git.checkout("-b", exp_ref)

    # Switch HEAD to auto/{branch}, including index but without modifying
    # the working tree
    repo.git.symbolic_ref("HEAD", f"refs/heads/{exp_ref}")
    repo.git.reset()

    # Add all the files and create a commit for the run
    repo.git.add("--all")
    commit_msg = f"Exp commit for {run}"
    repo.git.commit("--allow-empty", "-m", commit_msg)
    commit_sha = repo.head.object.hexsha

    # Switch back to the original state
    repo.git.symbolic_ref("HEAD", f"refs/heads/{cur_ref}")
    repo.git.reset()

    return commit_sha
