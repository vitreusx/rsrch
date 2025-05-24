"""A collection of various distributions for Torch. Basically a mirror of `torch.distributions`, except that the distributions are tensor-likes."""

from .affine import Affine
from .bernoulli import Bernoulli
from .beta import Beta
from .categorical import Categorical
from .clip_normal import ClipNormal
from .dirac import Dirac
from .discrete import Discrete, OneHot
from .distribution import Distribution
from .kl import kl_divergence, register_kl
from .mse_proxy import MSEProxy
from .normal import Normal
from .one_of import OneOf
from .tanh_normal import TanhNormal
from .transformed import Transformed
from .transforms import *
from .trunc_normal import TruncNormal
from .uniform import Uniform
