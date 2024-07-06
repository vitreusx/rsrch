from jax import *

del nn, random
from . import nn, random

PyTree = nn.Module
final = nn.final

from . import distributions
