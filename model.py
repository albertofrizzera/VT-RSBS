# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     28/08/2023
# -------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import neptune
import numpy as np


class CLIP_Projection(nn.Module):
    def __init__(self, output_size, input_size=512):
        super().__init__()
        self.projection = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.projection(x)
        return out