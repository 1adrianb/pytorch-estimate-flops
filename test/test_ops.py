import torch
import unittest
from pthflops import count_ops
from torchvision.models import resnet18


class Tester(unittest.TestCase):

    def test_overall(self):
        expected = 1826818048
        input = torch.rand(1, 3, 224, 224)
        net = resnet18()
        estimated = count_ops(net, input, print_readable=False, verbose=False)

        assert(expected == estimated)
