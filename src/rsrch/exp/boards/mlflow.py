import random
import tempfile
from numbers import Number
from pathlib import Path

import mlflow
from mlflow.utils.name_utils import _generate_random_name
from PIL import Image

from .base import Board, Step, StepMixin


class MLflow(StepMixin, Board):
    def __init__(self, exp_name: str):
        super().__init__()
        self.exp = mlflow.set_experiment(experiment_name=exp_name)

        # We temporarily revert to "true" RNG in order to make
        # experiment names different, even if the seed is the
        # same.
        rng_state = random.getstate()
        random.seed()
        run_name = _generate_random_name()
        random.setstate(rng_state)

        self.run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.exp.experiment_id,
        )

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        mlflow.log_metric(tag, value, self._get_step(step))

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        mlflow.log_image(image, key=tag, step=self._get_step(step))

    def add_video(self, tag, vid, *, step=None):
        with tempfile.TemporaryDirectory() as tmp_d:
            local_path = Path(tmp_d) / "video.mp4"
            vid.write_videofile(str(local_path))
            step_value = self._get_step(step)
            artifact_path = Path(tag) / f"step={step_value}.mp4"
            mlflow.log_artifact(str(local_path), str(artifact_path))
