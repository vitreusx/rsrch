from typing import Literal

import torch
import torchvision.transforms.functional as tv_F
from torch import Tensor, nn

from rsrch.torch.nn.utils import shape_infer_mode


class FCN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        in_channels: int,
        num_classes: int,
        output_stride: Literal[8, 16, 32] = 8,
        upsample: bool = False,
        interpolation_mode=tv_F.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_stride = output_stride
        self.interpolation_mode = interpolation_mode
        self.upsample = upsample

        # Check if encoder returns a list of tensors of proper sizes.
        # Use a tensor of size 224x224 as a dummy for testing purposes.
        # Also, extract feature channel counts for the sake of initializing
        #   conv layers.

        input_h = input_w = 224
        input_shape = in_channels, input_h, input_w
        dummy = torch.zeros((1, *input_shape))

        with shape_infer_mode(self.encoder):
            features: list[Tensor] = self.encoder(dummy)
            assert len(features) == 5
            feat_channels = []
            for fmap, stride in zip(features, (2, 4, 8, 16, 32)):
                fmap_h, fmap_w = input_h // stride, input_w // stride
                assert len(fmap.shape) == 4 and fmap.shape[2:] == (fmap_h, fmap_w)
                feat_channels.append(fmap.shape[1])

        self.proj_32 = nn.Conv2d(feat_channels[4], num_classes, 1, 1, 0)
        if output_stride < 32:
            self.proj_16 = nn.Conv2d(feat_channels[3], num_classes, 1, 1, 0)
        if output_stride < 16:
            self.proj_8 = nn.Conv2d(feat_channels[2], num_classes, 1, 1, 0)

    def forward(self, input: Tensor):
        features = self.encoder(input)
        feat8, feat16, feat32 = features[2:]

        # See Fig. 3 of the paper
        preds_32 = self.proj_32(feat32)
        if self.output_stride == 32:
            preds = preds_32
        else:
            preds_16 = self.proj_16(feat16)
            upscaled_32 = tv_F.resize(
                preds_32, preds_16.shape[-2:], self.interpolation_mode
            )
            preds_16 = preds_16 + upscaled_32
            if self.output_stride == 16:
                preds = preds_16
            else:
                preds_8 = self.proj_8(feat8)
                upscaled_16 = tv_F.resize(
                    preds_16, preds_8.shape[-2:], self.interpolation_mode
                )
                preds = preds_8 + upscaled_16

        if self.upsample:
            img_h, img_w = input.shape[-2:]
            preds = tv_F.resize(preds, (img_h, img_w), self.interpolation_mode)

        return preds
