from jax.lax import *

from rsrch.jax import amp

conv = amp.autocast_to_fp16(conv)
conv_general_dilated = amp.autocast_to_fp16(conv_general_dilated)
conv_transpose = amp.autocast_to_fp16(conv_transpose)
batch_matmul = amp.autocast_to_fp16(batch_matmul)

acos = amp.autocast_to_fp32(acos)
asin = amp.autocast_to_fp32(asin)
cosh = amp.autocast_to_fp32(cosh)
cumprod = amp.autocast_to_fp32(cumprod)
cumsum = amp.autocast_to_fp32(cumsum)
exp = amp.autocast_to_fp32(exp)
expm1 = amp.autocast_to_fp32(expm1)
log = amp.autocast_to_fp32(log)
log1p = amp.autocast_to_fp32(log1p)
pow = amp.autocast_to_fp32(pow)
reciprocal = amp.autocast_to_fp32(reciprocal)
sinh = amp.autocast_to_fp32(sinh)
tan = amp.autocast_to_fp32(tan)
erf_inv = amp.autocast_to_fp32(erf_inv)
sum = amp.autocast_to_fp32(sum)
rsqrt = amp.autocast_to_fp32(rsqrt)
