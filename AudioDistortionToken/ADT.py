
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class ADTLayer(nn.Module):
    def __init__(
            self, feature_size, num_audios,num_channels, anchor_size, kernel_size, stride
    ):
        super(ADTLayer, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=num_audios,
                out_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
        )

        self.filter_feat_size = int((feature_size - kernel_size) / stride + 1)
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=self.filter_feat_size, out_features=1),
                nn.Sigmoid()
            )
            for _ in range(num_channels)
        ])

        self.linear_out = nn.Sequential(
            nn.Linear(in_features=self.filter_feat_size, out_features=anchor_size),
            nn.ReLU(),
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x_pnn = self.conv1d(x)
        x_pnn_sum = torch.zeros_like(x_pnn[:, 0, :])
        for i in range(x_pnn.shape[1]):
            x_channel = x_pnn[:, i, :].reshape(batch_size, -1)
            w_channel = self.linears[i](x_channel)
            x_channel_weighted = x_channel * w_channel
            x_pnn_sum += x_channel_weighted
        x_anchor = self.linear_out(x_pnn_sum)
        return x_anchor

