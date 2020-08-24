import torch
import unittest
import pytest
from pthflops import count_ops
from torchvision.models import resnet18

# TODO: Add test for every op


class Tester(unittest.TestCase):

    def test_overall(self):
        expected = 1826843136
        input = torch.rand(1, 3, 224, 224)
        net = resnet18()
        estimated, estimations_dict = count_ops(net, input, print_readable=False, verbose=False)

        assert expected == pytest.approx(estimated, 1000000)
