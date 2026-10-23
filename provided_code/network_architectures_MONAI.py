from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import UNet

from provided_code.monai_data import N_INPUT_CHANNELS


class DosePredictionModel(nn.Module):
    """Small 3D U-Net baseline.
    THIS IS A GOOD PLACE FOR EXPERIMENTING
    """

    def __init__(self, 
                 in_channels: int = N_INPUT_CHANNELS,
                 spatial_dims : int= 3, 
                 channels: tuple[int] = (8, 16, 32, 64), # modest channel growth. notice, patch size must be dividable by 8! 
                 strides: tuple[int] = (2,2,2), 
                 num_res_units: int = 1,
                 norm: str = "INSTANCE"
                 ):
        super().__init__()
        self.network = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=1, #fixed, we only want a single dose channel
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dose is strictly positive. Softplus retains smooth gradients near zero.
        return F.softplus(self.network(x))
