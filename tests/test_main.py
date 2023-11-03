import unittest
import torch
from torch import nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import (
    CoSOV1, GConv, PConv,
    OutputLayer, CoSOV1Network
)
from data.transform import RGB2ChPairs

class CoSOV1Tests(unittest.TestCase):

    def test_RGB2ChPairs(self):
        input = torch.rand(size=(1, 3, 384, 384))
        module = RGB2ChPairs()
        out = module(input)
        self.assertEqual(out.shape, (1, 12, 384, 384))

    def test_GConv(self):
        x = torch.rand(1, 2, 384, 384)
        module = GConv(2, 2, kernel_size=3)
        output = module(x)
        self.assertEqual(output.shape, (1, 2, 384, 384))

    def test_PConv(self):
        x = torch.rand(1, 12, 384, 384)
        module = PConv(12, 12, kernel_size=1)
        output = module(x)
        self.assertEqual(output.shape, (1, 12, 384, 384))

    def test_CoSOV1(self):
        input = torch.rand(size=(1, 12, 384, 384))
        module = CoSOV1(12, 12, 6)
        out = module(input)
        self.assertEqual(out.shape, (1, 12, 384, 384))

    def test_CoSOV1Network(self):
        pass
    
    def test_output_layer(self):
        inputs = [torch.rand(1, 2, 384, 384) for _ in range(8)]
        module = OutputLayer()
        output = module(inputs)
        self.assertEqual(output.shape, (1, 1, 384, 384))

if __name__ == '__main__':
    unittest.main()