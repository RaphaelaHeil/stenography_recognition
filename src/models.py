from typing import Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t

from batchrenorm import BatchRenorm2d


class FullGatedConv2D(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, kernel_constraint=None):
        super().__init__(in_channels, out_channels * 2, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode,
                         device, dtype)
        self.kernel_constraint = kernel_constraint
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        if self.kernel_constraint:
            with torch.no_grad():
                self.weight = self.kernel_constraint(self.weight)
        output = self._conv_forward(input, self.weight, self.bias)
        return output[:, :self.out_channels, :, :] * nn.Sigmoid()(output[:, self.out_channels:, :, :])


class MaxNorm():

    def __init__(self, maxValue=2):
        self.maxValue = maxValue

    def __call__(self, weights):
        # norm = torch.linalg.vector_norm(dim=0, keepdim=True).clamp(min=self.maxValue / 2)
        norm = weights.norm(2, dim=0, keepdim=True).clamp(min=self.maxValue / 2)
        desired = torch.clamp(norm, max=self.maxValue)
        weights *= (desired / norm)
        return weights


class FlorDecoder(nn.Module):

    def __init__(self, alphabetSize: int):
        super().__init__()
        self.gru1 = nn.GRU(input_size=64, hidden_size=128, bidirectional=True)
        self.gru1Dropout = nn.Dropout(0.5)
        self.gru2 = nn.Sequential(
                nn.Linear(in_features=256, out_features=256),
                nn.GRU(input_size=256, hidden_size=128, bidirectional=True)
        )
        self.gru2Dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=256, out_features=alphabetSize)

    def forward(self, features):
        encoding1, _ = self.gru1(features)
        encoding1 = self.gru1Dropout(encoding1)
        encoding2, _ = self.gru2(encoding1)
        encoding2 = self.gru2Dropout(encoding2)
        return self.linear(encoding2)


class Flor(nn.Module):

    def __init__(self, alphabetSize: int):
        super().__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.PReLU(num_parameters=16),
                BatchRenorm2d(16),
                FullGatedConv2D(in_channels=16, out_channels=16, kernel_size=(3, 3), padding="same"),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"),
                nn.PReLU(num_parameters=32),
                BatchRenorm2d(32),
                FullGatedConv2D(in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"),

                nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(4, 2), stride=(4, 2), padding=(2, 1)),
                nn.PReLU(num_parameters=40),
                BatchRenorm2d(40),
                FullGatedConv2D(in_channels=40, out_channels=40, kernel_size=(3, 3), padding="same",
                                kernel_constraint=MaxNorm(4)),
                nn.Dropout(0.2),

                nn.Conv2d(in_channels=40, out_channels=48, kernel_size=(3, 3), padding="same"),
                nn.PReLU(num_parameters=48),
                BatchRenorm2d(48),
                FullGatedConv2D(in_channels=48, out_channels=48, kernel_size=(3, 3), padding="same",
                                kernel_constraint=MaxNorm(4)),
                nn.Dropout(0.2),
                #
                nn.Conv2d(in_channels=48, out_channels=56, kernel_size=(4, 2), stride=(4, 2), padding=(2, 1)),
                nn.PReLU(num_parameters=56),
                BatchRenorm2d(56),
                FullGatedConv2D(in_channels=56, out_channels=56, kernel_size=(3, 3), padding="same",
                                kernel_constraint=MaxNorm(4)),
                nn.Dropout(0.2),

                nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.PReLU(num_parameters=64),
                BatchRenorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.head = FlorDecoder(alphabetSize)

    def forward(self, input):
        features = self.cnn(input)
        # features = features.permute(3, 0, 1, 2)  # (width, batch, channel, height)
        # features = features.reshape(features.shape[0], -1, 64)  # TODO: double-check with batch!
        features = features.permute(2, 3, 0, 1)
        features = features.reshape(-1, features.shape[2], 64)

        return self.head(features)


class GatedBN(nn.Module):

    def __init__(self, alphabetSize: int):
        super().__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.PReLU(num_parameters=16),
                nn.BatchNorm2d(16),
                FullGatedConv2D(in_channels=16, out_channels=16, kernel_size=(3, 3), padding="same"),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"),
                nn.PReLU(num_parameters=32),
                nn.BatchNorm2d(32),
                FullGatedConv2D(in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"),

                nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(4, 2), stride=(4, 2), padding=(2, 1)),
                nn.PReLU(num_parameters=40),
                nn.BatchNorm2d(40),
                FullGatedConv2D(in_channels=40, out_channels=40, kernel_size=(3, 3), padding="same",
                                kernel_constraint=MaxNorm(4)),
                nn.Dropout(0.2),

                nn.Conv2d(in_channels=40, out_channels=48, kernel_size=(3, 3), padding="same"),
                nn.PReLU(num_parameters=48),
                nn.BatchNorm2d(48),
                FullGatedConv2D(in_channels=48, out_channels=48, kernel_size=(3, 3), padding="same",
                                kernel_constraint=MaxNorm(4)),
                nn.Dropout(0.2),
                #
                nn.Conv2d(in_channels=48, out_channels=56, kernel_size=(4, 2), stride=(4, 2), padding=(2, 1)),
                nn.PReLU(num_parameters=56),
                nn.BatchNorm2d(56),
                FullGatedConv2D(in_channels=56, out_channels=56, kernel_size=(3, 3), padding="same",
                                kernel_constraint=MaxNorm(4)),
                nn.Dropout(0.2),

                nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(3, 3), padding="same"),
                nn.PReLU(num_parameters=64),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.head = FlorDecoder(alphabetSize)

    def forward(self, input):
        features = self.cnn(input)

        n, c, h, w = features.size()

        features = features.permute(3, 0, 1, 2)  # (width, batch, channel, height)
        features = features.reshape(w, n, h * c)

        return self.head(features)