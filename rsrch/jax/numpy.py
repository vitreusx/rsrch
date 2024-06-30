from jax.numpy import *

from rsrch.jax import amp

matmul = amp.autocast_to_fp16(matmul)

acos = amp.autocast_to_fp32(acos)
asin = amp.autocast_to_fp32(asin)
cosh = amp.autocast_to_fp32(cosh)
cumprod = amp.autocast_to_fp32(cumprod)
cumsum = amp.autocast_to_fp32(cumsum)
exp = amp.autocast_to_fp32(exp)
expm1 = amp.autocast_to_fp32(expm1)
log = amp.autocast_to_fp32(log)
log10 = amp.autocast_to_fp32(log10)
log1p = amp.autocast_to_fp32(log1p)
log2 = amp.autocast_to_fp32(log1p)
pow = amp.autocast_to_fp32(pow)
prod = amp.autocast_to_fp32(prod)
reciprocal = amp.autocast_to_fp32(reciprocal)
sinh = amp.autocast_to_fp32(sinh)
sum = amp.autocast_to_fp32(sum)
tan = amp.autocast_to_fp32(tan)
