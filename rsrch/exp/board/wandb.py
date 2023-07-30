import PIL.Image
import torchvision.transforms.functional as F
import wandb

from rsrch.exp.board.api import VideoLike

from . import api


class Wandb(api.Board):
    def __init__(self, *, project: str, step_fn=None):
        self.run = wandb.init(project=project)
        self.step_fn = step_fn
        self._scalars = {}
        self._samples = {}
        self._images = {}

    def _get_step(self, step):
        if step is None:
            step = self.step_fn()
        return step

    def add_scalar(self, tag, value, *, step=None):
        step = self._get_step(step)

        if tag not in self._scalars:
            data = {"step": f"{tag}__step"}
            self._scalars[tag] = data
            self.run.define_metric(data["step"])
            self.run.define_metric(tag, step_metric=data["step"])
            self._scalars.add(tag)
        data = self._scalars[tag]

        self.run.log({tag: value, data["step"]: step})

    def add_samples(self, tag, values, *, step=None):
        step = self._get_step(step)

        if tag not in self._samples:
            xs = []
            ys = [[] for _ in values]
            self._samples[tag] = {"xs": xs, "ys": ys}

        data = self._samples[tag]
        data["xs"].append(tag)
        for ys, val in zip(data["ys"], values):
            ys.append(val)

        self.run.log({tag: wandb.plot.line_series(data["xs"], data["ys"])})

    def add_image(self, tag, image, *, step=None):
        step = self._get_step(step)
        if tag not in self._images:
            data = {"step": f"{tag}__step"}
            self._images[tag] = data
            self.run.define_metric(data["step"])
            self.run.define_metric(tag, step_metric=data["step"])
            self._images.add(tag)
        data = self._images[tag]

        if not isinstance(image, PIL.Image.Image):
            image = F.to_pil_image(image)
        self.run.log({tag: wandb.Image(image), data["step"]: step})

    def add_video(self, tag, video, *, fps, step):
        raise NotImplementedError
