import math
from functools import partial

import torch.nn.functional as F
from torch import Tensor, nn

from rsrch import spaces
from rsrch.exp import Experiment, board
from rsrch.nn import dh


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10 * 2)


def main():
    exp = Experiment(project="vae")
    exp.add_board(board.WeightsAndBiases(dir=exp.dir, project="vae"))


if __name__ == "__main__":
    main()
