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