import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# definition of the leaky hardtanh function
def leaky_hardtanh(input, min_val=- 1.0, max_val=1.0, min_slope=0.01, max_slope=0.01):
    '''
        Defines the Leaky Hardtanh function
    '''
    return torch.where(input < max_val, F.leaky_relu(input-min_val, min_slope)+min_val, (input-max_val)*max_slope+max_val)

  
# create a class wrapper from PyTorch nn.Module
class LeakyHardtanh(nn.Module):
    r"""Applies the Leaky HardTanh function element-wise.

    Leaky HardTanh is defined as:

    .. math::
        \text{LeakyHardTanh}(x) = \begin{cases}
            (x - \text{max\_val}) \ times x +  \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            (x - \text{min\_val}) \ times x +  \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        min_slope: Controls the angle of the region below min_val. Default: 0.01
        max_slope: Controls the angle of the region above max_val. Default: 0.01

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.LeakyHardtanh(-2, 2, 0.001, 0.001)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, min_val: float =- 1.0, max_val: float=1.0, min_slope: float=0.01, max_slope: float=0.01):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.min_slope = min_slope
        self.max_slope = max_slope
        assert self.max_val > self.min_val, "max_val must be larger than min_val"

    def forward(self, input: Tensor) -> Tensor:
        return leaky_hardtanh(input, self.min_val, self.max_val, self.min_slope, self.max_slope)
