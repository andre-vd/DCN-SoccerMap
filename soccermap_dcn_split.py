from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from torchvision.ops import DeformConv2d

""" SoccerMap Implementation from
https://github.com/ML-KULeuven/un-xPass/blob/main/unxpass/components/soccermap.py

"""

class _DeformableFeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.symmetric_padding = nn.ReplicationPad2d((2, 2, 2, 2))

        self.offset_conv1 = nn.Conv2d(
            in_channels, 50, kernel_size=5, stride=1, padding=0
        )
        self.deform_conv1 = DeformConv2d(
            in_channels, out_channels // 2, kernel_size=5, stride=1, padding=0
        )

        self.offset_conv2 = nn.Conv2d(
            out_channels // 2, 50, kernel_size=5, stride=1, padding=0
        )
        self.deform_conv2 = DeformConv2d(
            out_channels // 2, out_channels, kernel_size=5, stride=1, padding=0
        )

    def forward(self, x):
        x_padded = self.symmetric_padding(x)

        offset1 = self.offset_conv1(x_padded)
        x = F.relu(self.deform_conv1(x_padded, offset1))

        
        x_padded = self.symmetric_padding(x)

        offset2 = self.offset_conv2(x_padded)
        x = F.relu(self.deform_conv2(x_padded, offset2))

        return x

class _FeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(5, 5), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=(5, 5), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((2, 2, 2, 2))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)
        return x


class _PredictionLayer(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)  # linear activation
        return x


class _UpSamplingLayer(nn.Module):
    """The upsampling layer of the SoccerMap architecture.

    The upsampling layer provides non-linear upsampling by first applying a 2x
    nearest neighbor upsampling and then two layers of convolutional filters.
    The first convolutional layer consists of 32 filters with a 3x3 activation
    field and stride 1, followed by a ReLu activation layer. The second layer
    consists of 1 layer with a 3x3 activation field and stride 1, followed by
    a linear activation layer. This upsampling strategy has been shown to
    provide smoother outputs.
    """

    def __init__(self):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = self.symmetric_padding(x)
        x = self.conv2(x)  # linear activation
        x = self.symmetric_padding(x)
        return x


class _FusionLayer(nn.Module):
    """The fusion layer of the SoccerMap architecture.

    The fusion layer merges the final prediction surfaces at different scales
    to produce a final prediction. It concatenates the pair of matrices and
    passes them through a convolutional layer of one 1x1 filter.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1)

    def forward(self, x: List[torch.Tensor]):
        out = self.conv(torch.cat(x, dim=1))  # linear activation
        return out


class SoccerMapDCNMixed(nn.Module):
    """SoccerMap architecture with mixed deformable and non-deformable feature extraction."""

    def __init__(self, in_channels):
        super().__init__()

        # Feature extraction layers for the initial scale
        self.features_x1_deform = _FeatureExtractionLayer(8, out_channels=32)
        self.features_x1_non_deform = _FeatureExtractionLayer(5, out_channels=32)

        # Downsampling layers
        self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))

        # Feature extraction layers for the second scale
        # Adjusted input channels to match the concatenated output channels
        self.features_x2_deform = _DeformableFeatureExtractionLayer(32, out_channels=32)
        self.features_x2_non_deform = _FeatureExtractionLayer(32, out_channels=32)

        # Feature extraction layer for the third scale
        self.features_x4 = _FeatureExtractionLayer(64, out_channels=64)

        # Upsampling and fusion layers
        self.up_x2 = _UpSamplingLayer()
        self.up_x4 = _UpSamplingLayer()
        self.fusion_x2_x4 = _FusionLayer()
        self.fusion_x1_x2 = _FusionLayer()

        # Prediction layers
        self.prediction_x1 = _PredictionLayer(in_channels=64)
        self.prediction_x2 = _PredictionLayer(in_channels=64)
        self.prediction_x4 = _PredictionLayer(in_channels=64)

    def forward(self, x):
        # Split the input tensor into deformable and non-deformable parts
        x_deform = x[:, :8, :, :]    
        x_non_deform = x[:, 8:, :, :]  

        f_x1_deform = self.features_x1_deform(x_deform)
        f_x1_non_deform = self.features_x1_non_deform(x_non_deform)
        f_x1 = torch.cat([f_x1_deform, f_x1_non_deform], dim=1)  # Shape: (batch_size, 64, H, W)

        # Downsample the combined features
        f_x1_down = self.down_x2(f_x1)

        f_x1_down_deform = f_x1_down[:, :32, :, :]     # First 32 channels
        f_x1_down_non_deform = f_x1_down[:, 32:, :, :] # Next 32 channels

        f_x2_deform = self.features_x2_deform(f_x1_down_deform)
        f_x2_non_deform = self.features_x2_non_deform(f_x1_down_non_deform)

        f_x2 = torch.cat([f_x2_deform, f_x2_non_deform], dim=1)  # Shape: (batch_size, 64, H/2, W/2)
        f_x2_down = self.down_x4(f_x2)

        f_x4 = self.features_x4(f_x2_down)  # Input channels adjusted to 64

        # Prediction at each scale
        pred_x1 = self.prediction_x1(f_x1)
        pred_x2 = self.prediction_x2(f_x2)
        pred_x4 = self.prediction_x4(f_x4)

        # Fusion of predictions from different scales
        x4x2combined = self.fusion_x2_x4([self.up_x4(pred_x4), pred_x2])
        combined = self.fusion_x1_x2([self.up_x2(x4x2combined), pred_x1])

        # Final output with sigmoid activation
        output = torch.sigmoid(combined)
        return output