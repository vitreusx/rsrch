from rsrch.rl.data.step import Step, TensorStep
from rsrch.rl.data.trajectory import TensorTrajectory, Trajectory


def to_tensor_step(step: Step) -> TensorStep:
    return TensorStep.convert(step)


def to_tensor_seq(seq: Trajectory) -> TensorTrajectory:
    return TensorTrajectory.convert(seq)
