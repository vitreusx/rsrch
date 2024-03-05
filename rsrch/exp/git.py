from pathlib import Path

import git


def create_auto_commit(run: str):
    """Create an auto-commit for a run in a separate branch named
    `auto/{branch}`, where {branch} is the current branch name."""

    repo = git.Repo(Path(__file__).parent, search_parent_directories=True)

    # Get current branch name
    branch = repo.git.rev_parse("--abbrev-ref", "HEAD")
    auto = f"auto/{branch}"

    # Create auto/{branch} if necessary
    if not any(ref.name == auto for ref in repo.refs):
        repo.git.checkout("-b", auto)

    # Switch HEAD to auto/{branch}, including index but without modifying
    # the working tree
    repo.git.symbolic_ref("HEAD", f"refs/heads/{auto}")
    repo.git.reset()

    # Add all the files and create a commit for the run
    repo.git.add("--all")
    commit_msg = f"[Auto] Commit for {run}"
    repo.git.commit("--allow-empty", "-m", commit_msg)

    # Switch back to the original state
    repo.git.symbolic_ref("HEAD", f"refs/heads/{branch}")
    repo.git.reset()
