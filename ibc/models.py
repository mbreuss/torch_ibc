import dataclasses
import enum
from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CoordConv, GlobalAvgPool2d, GlobalMaxPool2d, SpatialSoftArgmax


class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU


@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU


class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        dropout_layer: Callable
        if config.dropout_prob is not None:
            dropout_layer = partial(nn.Dropout, p=config.dropout_prob)
        else:
            dropout_layer = nn.Identity

        layers: Sequence[nn.Module]
        if config.hidden_depth == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                nn.Linear(config.input_dim, config.hidden_dim),
                config.activation_fn.value(),
                dropout_layer(),
            ]
            for _ in range(config.hidden_depth - 1):
                layers += [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    config.activation_fn.value(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config.hidden_dim, config.output_dim)]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn.value()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    blocks: Sequence[int] = dataclasses.field(default=(16, 32, 32))
    activation_fn: ActivationType = ActivationType.RELU


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        depth_in = config.in_channels

        layers = []
        for depth_out in config.blocks:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    ResidualBlock(depth_out, config.activation_fn),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = config.activation_fn.value()
        print('SmallCNN parameters: {}'.format(sum(p.numel() for p in self.net.parameters())))

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d
    MAX_POOL = GlobalMaxPool2d


@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.AVERAGE_POOL
    coord_conv: bool = False


class ConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        out = self.mlp(out)
        return out


class EBMConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)
        print('single CNN parameters: {}'.format(sum(p.numel() for p in self.conv.parameters())))
        print('reducer: {}'.format(sum(p.numel() for p in self.reducer.parameters())))

        print('CNN parameters: {}'.format(sum(p.numel() for p in self.cnn.parameters()) +
                                       sum(p.numel() for p in self.conv.parameters()) +
                                       sum(p.numel() for p in self.reducer.parameters())))
        print('mlp parameters: {}'.format(sum(p.numel() for p in self.mlp.parameters())))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)


if __name__ == "__main__":
    config = ConvMLPConfig(
        cnn_config=CNNConfig(5),
        mlp_config=MLPConfig(32, 128, 2, 2),
        spatial_reduction=SpatialReduction.AVERAGE_POOL,
        coord_conv=True,
    )

    net = ConvMLP(config)
    print(net)

    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        out = net(x)
    print(out.shape)
