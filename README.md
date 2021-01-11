[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  [![Test Pytorch Flops Counter](https://github.com/1adrianb/pytorch-estimate-flops/workflows/Test%20Pytorch%20Flops%20Counter/badge.svg)](https://travis-ci.com/1adrianb/pytorch-estimate-flops)
[![PyPI](https://img.shields.io/pypi/v/pthflops.svg?style=flat)](https://pypi.org/project/pthflops/)

# pytorch-estimate-flops

Simple pytorch utility that estimates the number of FLOPs for a given network. For now only some basic operations are supported (basically the ones I needed for my models). More will be added soon.

All contributions are welcomed.

## Installation

You can install the model using pip:

```bash
pip install pthflops
```
or directly from the github repository:
```bash
git clone https://github.com/1adrianb/pytorch-estimate-flops && pytorch-estimate-flops
python setup.py install
```

## Example

```python
import torch
from torchvision.models import resnet18

from pthflops import count_ops

# Create a network and a corresponding input
device = 'cuda:0'
model = resnet18().to(device)
inp = torch.rand(1,3,224,224).to(device)

# Count the number of FLOPs
count_ops(model, inp)
```

Ignoring certain layers:

```python
import torch
from torch import nn
from pthflops import count_ops

class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.conv1 = nn.Conv2d(5, 5, 1, 1, 0)
        # ... other layers present inside will also be ignored

    def forward(self, x):
        return self.conv1(x)

# Create a network and a corresponding input
inp = torch.rand(1,5,7,7)
net = nn.Sequential(
    nn.Conv2d(5, 5, 1, 1, 0),
    nn.ReLU(inplace=True),
    CustomLayer()
)

# Count the number of FLOPs
count_ops(net, inp, ignore_layers=['CustomLayer'])

```
