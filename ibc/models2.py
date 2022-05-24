import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ibc.modules import CoordConv, GlobalAvgPool2d, GlobalMaxPool2d, SpatialSoftArgmax
from omegaconf import DictConfig
import hydra

from typing import Optional, Tuple


def load_spatial_module(module: str):
    if module == ' GlobalAvgPool2d':
        model = GlobalAvgPool2d()
    elif module == 'GlobalMaxPool2d':
        model = GlobalMaxPool2d()
    elif module == 'SpatialSoftArgmax':
        model = SpatialSoftArgmax()
    else:
        ValueError('Module is not implemented! Please check spelling.')
    return model


def return_activiation_fcn(activation_type: str):
    # build the activation layer
    if activation_type == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif activation_type == 'tanh':
        act = torch.nn.Sigmoid()
    elif activation_type == 'ReLU':
        act = torch.nn.ReLU()
    elif activation_type == 'PReLU':
        act = torch.nn.PReLU()
    elif activation_type == 'softmax':
        act = torch.nn.Softmax(dim=-1)
    else:
        act = torch.nn.PReLU()
    return act


class MLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network for benchmarking the performance of different networks. The model is used in
    several papers and can be used to compare all_plots model performances.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 100,
                 num_hidden_layers: int = 1,
                 output_dim=1,
                 dropout: int = 0.25,
                 activation: str = 'sigmoid'):
        super(MLPNetwork, self).__init__()
        self.network_type = 'mlp'
        # define number of variables in an input sequence
        self.input_dim = input_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = output_dim
        self.dropout = dropout
        # set up the network
        # create dict with all input keys:
        self.init_dict = {'input_dim': self.input_dim,
                          'hidden_dim': self.hidden_dim,
                          'num_hidden_layers': self.num_hidden_layers,
                          'output_dim': self.output_dim,
                          'activation': activation,
                          'dropou': dropout
                          }
        # stack the desired number of hidden layers
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        self.layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(1, self.num_hidden_layers)])
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # build the activation layer
        self.act = return_activiation_fcn(activation)

    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.act(x)
        return x

    def get_device(self, device: torch.device):
        self._device = device


class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: bool = 'ReLU',
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        # build the activation layer
        self.act = return_activiation_fcn(activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO Fix mistakes here does not make sense lol
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + x


class SmallCNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self,
                 input_width: int,
                 input_height: int,
                 in_channels: int,
                 blocks: DictConfig,
                 activation_fn: str,
                 output_dimension: int,
                 linear_dropout: float,
                 l2_normalize_output: bool
                 )-> None:
        super().__init__()

        self._depth_in = in_channels
        self._blocks = blocks
        self._output_dim = output_dimension
        self._linear_dropout = linear_dropout
        self._l2_normalize_output = l2_normalize_output
        w, h = self.calc_out_size(input_width, input_height, 3, 1, 1)
        w, h = self.calc_out_size(w, h, 3, 1, 1)
        w, h = self.calc_out_size(w, h, 3, 1, 1)
        w, h = self.calc_out_size(w, h, 1, 1, 1)
        self.act = return_activiation_fcn(activation_fn)

        self.spatial_softmax = SpatialSoftmax(num_rows=w, num_cols=h, temperature=1.0)
        layers = []
        for depth_out in blocks[:-1]:
            layers.extend(
                [
                    nn.Conv2d(in_channels=self._depth_in, out_channels=depth_out, kernel_size=3, padding=1),
                    ResidualBlock(depth_out, activation_fn)
                ]
            )
            self._depth_in = depth_out
        layers.extend(
            [
                self.act,
                nn.Conv2d(in_channels=self._depth_in, out_channels= blocks[-1], kernel_size=1, padding=1)
            ]
        )

        self.net = nn.Sequential(*layers)
        print('SmallCNN parameters: {}'.format(sum(p.numel() for p in self.net.parameters())))
        print('softmax parameters: {}'.format(sum(p.numel() for p in self.spatial_softmax.parameters())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = self.act(out)
        out = self.spatial_softmax(out)
        return out

    def get_device(self, device: torch.device):
        self._device = device

    @staticmethod
    def calc_out_size(w: int, h: int, kernel_size: int, padding: int, stride: int) -> Tuple[int, int]:
        width = (w - kernel_size + 2 * padding) // stride + 1
        height = (h - kernel_size + 2 * padding) // stride + 1
        return width, height


class ConvMLP(nn.Module):
    def __init__(self,
                 small_cnn: DictConfig,
                 mlp: DictConfig,
                 coord_conv: bool = True) -> None:
        super().__init__()

        self.coord_conv = coord_conv
        self.cnn = hydra.utils.instantiate(small_cnn)
        self.mlp = hydra.utils.instantiate(mlp)

    def get_device(self, device: torch.device):
        self._device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x)
        out = self.mlp(out)
        return out


class EBMConvMLP(nn.Module):
    def __init__(self,
                 small_cnn: DictConfig,
                 mlp: DictConfig,
                 coord_conv: bool = False) -> None:
        super().__init__()

        self.coord_conv = coord_conv
        if coord_conv:
            small_cnn.in_channels = small_cnn.in_channels + 2
        self.cnn = hydra.utils.instantiate(small_cnn)
        self.mlp = hydra.utils.instantiate(mlp)

        print('CNN parameters: {}'.format(sum(p.numel() for p in self.cnn.parameters())))
        print('mlp parameters: {}'.format(sum(p.numel() for p in self.mlp.parameters())))

    def get_device(self, device: torch.device):
        self._device = device
        self.cnn.get_device(device)
        self.mlp.get_device(device)
        self.cnn.to(device)
        self.mlp.to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self._device) # [B, D, V1, V2]
        y = y.to(self._device) # [B, N, dim_samples]
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x) # [B, 32]
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1) #
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        # fused = torch.cat([out, y], dim=1)
        out = self.mlp(fused)
        return out.view(B, N)


# from https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/models/perceptual_encoders/vision_network.py
class SpatialSoftmax(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, temperature: Optional[float] = None):
        """
        Computes the spatial softmax of a convolutional feature map.
        Read more here:
        "Learning visual feature spaces for robotic manipulation with
        deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
        :param num_rows:  size related to original image width
        :param num_cols:  size related to original image height
        :param temperature: Softmax temperature (optional). If None, a learnable temperature is created.
        """
        super(SpatialSoftmax, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, num_cols), torch.linspace(-1.0, 1.0, num_rows), indexing="ij"
        )
        x_map = grid_x.reshape(-1)
        y_map = grid_y.reshape(-1)
        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)
        if temperature:
            self.register_buffer("temperature", torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(-1, h * w)  # batch, C, W*H
        softmax_attention = F.softmax(x / self.temperature, dim=1)  # batch, C, W*H
        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        self.coords = expected_xy.view(-1, c * 2)
        return self.coords  # batch, C*2


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    net = hydra.utils.instantiate(cfg.agent.model)
    # net = ConvMLP(cfg)
    print(net)

    x = torch.randn(2, 3, 96, 96)
    y = torch.rand(2, 2)
    with torch.no_grad():
        out = net(x, y)
    print(out.shape)


if __name__ == "__main__":

    main()