from copy import copy

import numpy as np
import torch
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch import Tensor

from rsrch.data.imagenet import ImageNet
from rsrch.rl import data as rl
from rsrch.rl.data import Buffer, BufferData, SliceBatch, StepBatch
from rsrch.utils import data
from rsrch.utils.data import *


def to_pil(obs: Tensor):
    img = obs.float().cpu().clamp(0.0, 1.0)
    return tv_F.to_pil_image(img)


class Observations(data.Dataset):
    def __init__(self, buffer: rl.Buffer):
        super().__init__()
        self.buffer = buffer
        self.ts_view = rl.TimestepView(self.buffer)

    def __len__(self):
        return len(self.ts_view)

    def __getitem__(self, idx):
        obs = self.ts_view[self.ts_view.ids[idx]]
        obs = self._obs_t(obs)
        return obs

    def _obs_t(self, obs):
        if self.buffer.stack_num is not None:
            obs = obs[-1]
        obs = torch.as_tensor(obs / 255.0, dtype=torch.float32)
        obs = tv_F.resize(obs, (64, 64))
        return obs

    @staticmethod
    def collate_fn(batch: list):
        return torch.stack(batch)


class Steps(data.Dataset):
    def __init__(self, buffer: rl.Buffer):
        super().__init__()
        self.buffer = buffer
        self.step_view = rl.StepView(self.buffer)

    def __len__(self):
        return len(self.step_view)

    def __getitem__(self, idx):
        step_id = self.step_view.ids[idx]
        step = self.step_view[step_id]
        step.obs = self._obs_t(step.obs)
        step.next_obs = self._obs_t(step.next_obs)
        return idx, step

    def _obs_t(self, obs):
        if self.buffer.stack_num is not None:
            obs = obs[-1]
        obs = torch.as_tensor(obs / 255.0, dtype=torch.float32)
        obs = tv_F.resize(obs, (64, 64))
        return obs

    @staticmethod
    def collate_fn(batch: list):
        return rl.default_collate_fn(batch)


class Slices(data.Dataset):
    def __init__(self, buffer: rl.Buffer, slice_steps: int):
        super().__init__()
        self.buffer = buffer
        self.slice_view = rl.SliceView(self.buffer, slice_steps)

    def __len__(self):
        return len(self.slice_view)

    def __getitem__(self, idx):
        slice_id = self.slice_view.ids[idx]
        slice = self.slice_view[slice_id]
        slice.obs = self._obs_t(slice.obs)
        return slice

    def _obs_t(self, obs):
        obs = np.stack(obs)
        obs = torch.as_tensor(obs / 255.0, dtype=torch.float32)
        if self.buffer.stack_num is not None:
            obs = obs.flatten(1, 2)
        obs = tv_F.resize(obs, (64, 64))
        return obs

    @staticmethod
    def collate_fn(batch: list):
        return rl.default_collate_fn(batch)


class SlicesLM(data.Dataset):
    def __init__(self, buffer: rl.Buffer, slice_steps: int):
        super().__init__()
        self.buffer = buffer
        self.slice_view = rl.SliceView(self.buffer, slice_steps)

    def __len__(self):
        return len(self.slice_view)

    def __getitem__(self, idx):
        slice_id = self.slice_view.ids[idx]
        slice = self.slice_view[slice_id]
        return slice

    def to_pil(self, obs: Tensor):
        img = obs.float().cpu().clamp(0.0, 1.0)
        return tv_F.to_pil_image(img)

    @staticmethod
    def collate_fn(batch: list):
        return rl.default_collate_fn(batch)


class Episodes(data.Dataset):
    def __init__(self, buffer: rl.Buffer):
        super().__init__()
        self.buffer = buffer
        self.ep_view = rl.EpisodeView(self.buffer)

    def __len__(self):
        return len(self.ep_view)

    def __getitem__(self, idx):
        ep_id = self.ep_view.ids[idx]
        ep = self.ep_view[ep_id]
        ep.obs = self._obs_t(ep.obs)
        ep.act = self._act_t(ep.act)
        ep.reward = torch.as_tensor(ep.reward)
        return ep

    def _obs_t(self, obs):
        obs = np.stack(obs)
        if self.buffer.stack_num is not None:
            obs = obs[:, -1]
        obs = torch.as_tensor(obs / 255.0, dtype=torch.float32)
        obs = tv_F.resize(obs, (64, 64))
        return obs

    def _act_t(self, act):
        if isinstance(act[0], np.ndarray):
            act = np.stack(act)
        else:
            act = np.array(act)
        return torch.from_numpy(act)

    @staticmethod
    def collate_fn(batch: list):
        return rl.default_collate_fn(batch)
