import torch
import torch.nn as nn

class GroupwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(GroupwiseConv2d, self).__init__()
        self.groups = groups
        self.convs = nn.ModuleList()
        for i in range(groups):
            self.convs.append(nn.Conv2d(in_channels // groups, out_channels // groups, kernel_size))

    def forward(self, x):
        x_split = torch.split(x, x.size(1) // self.groups, dim=1)
        out = []
        for i in range(self.groups):
            out.append(self.convs[i](x_split[i]))
        return torch.cat(out, dim=1)

class ResidualCoSOV1(nn.Module)
    def __init__(self):
        super(ResidualCoSOV1, self).__init__()
        pass

    def forward(self, x):
        pass


class EncoderUnit(nn.Module)
    def __init__(self):
        super(EncoderUnit, self).__init__()
        pass

    def forward(self, x):
        pass

def test_GroupwiseConv2d():
    x = torch.randn(1, 12, 28, 28)
    conv = GroupwiseConv2d(12, 24, 3, 4)
    out = conv(x)
    print(out.shape)

if __name__ == "__main__":
    test_GroupwiseConv2d()