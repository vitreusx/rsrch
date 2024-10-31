# Source code for _Investigating and Combining Approaches for Data-Efficient Model-based RL_

This repository contains the source code for MSc thesis _Investigating and Combining Approaches for Data-Efficient Model-based RL_.

## Contents

The repo consists of:

- a Python package [`rsrch`](./rsrch/) containing the code for launching experiments;
- [thesis-related files](./thesis/), including:
  - [TeX files](./thesis/tex) and [the compiled pdf](./thesis/tex/thesis.pdf);
  - [Jupyter notebooks](./thesis/analysis/) with data analysis and the relevant plots in the thesis.
  - [experiment presets and launchers](./thesis/exp)

## Requirements

- **OS**: Linux (x64)
- **Hardware**: Modern NVIDIA GPU
- **Software**: Git, Python 3.10, [Poetry](https://python-poetry.org/)
  - If using the Ansible playbook, these are installed automatically.

## Installation

### Dev setup

Clone the repository and install the package using Poetry:

```shell
git clone https://github.com/vitreusx/rsrch
cd rsrch
poetry install --with dev
```

### Setup on a remote host

An Ansible playbook [`rsrch/launch/setup_hosts.yml`](./rsrch/launch/setup_hosts.yml) is provided, which one can use to automatically clone and set up the repo on a remote host.

## Launching experiments via Slurm

The scripts in [`thesis/exp`](./thesis/exp) each output a list of regular shell commands to execute. To launch these, one ought to save these to a text file (e.g. `commands.txt` at the root of the repo) and use `python -m rsrch.launch.via_slurm`.
