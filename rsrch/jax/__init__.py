from jax import *

del numpy, lax, nn, random
from . import lax, nn, numpy, random
from .amp import autocast

PyTree = nn.Module
final = nn.final
