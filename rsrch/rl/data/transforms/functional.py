from rsrch.rl.data.seq import Sequence, TensorSeq
from rsrch.rl.data.step import Step, TensorStep


def to_tensor_step(step: Step) -> TensorStep:
    return TensorStep.convert(step)


def to_tensor_seq(seq: Sequence) -> TensorSeq:
    return TensorSeq.convert(seq)
