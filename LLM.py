import math
import inspect
from dataclasses import dataclass
import torch as t
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module): #normalize layer with bias = False
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(t.ones(ndim))
        self.bias = nn.Parameter(t.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

