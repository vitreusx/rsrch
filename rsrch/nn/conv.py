import torch
import torch.nn as nn


class ConvNet(nn.Sequential):
    def __init__(
        self,
        *,
        in_channels: int = None,
        input_shape: torch.Size = None,
        depth=4,
        kernel_size=3,
        hidden_dim=64,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        flatten=True,
    ):
        layers = []

        self.out_features = None
        if input_shape is not None:
            c, h, w = input_shape[-3:]
            scale = 2**depth
            assert h % scale == 0 and w % scale == 0
            num_features = [c, *((2**idx) * hidden_dim for idx in range(depth))]
            final_size = torch.Size([num_features[-1], h // scale, w // scale])
        elif in_channels is not None:
            c = in_channels
            num_features = [c, *((2**idx) * hidden_dim for idx in range(depth))]

        p = kernel_size // 2
        for in_feat, out_feat in zip(num_features[:-1], num_features[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_feat, out_feat, kernel_size, padding=p),
                    norm_layer(out_feat),
                    act_layer(),
                    nn.MaxPool2d(2),
                ]
            )

        if self.out_features is not None and flatten:
            self.out_features = final_size.numel()
            layers.append(nn.Flatten())

        super().__init__(*layers)
