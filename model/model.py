import torch
from torch import nn
from torchvision.ops import DropBlock2d
    
class GConv(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super().__init__()
        self.in_channels = kwargs.get('in_channels', in_channels)
        self.out_channels = kwargs.get('out_channels', out_channels)
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.padding = kwargs.get('padding', 1)
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.elu(out)
        return out

class PConv(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super().__init__()
        self.in_channels = kwargs.get('in_channels', in_channels)
        self.out_channels = kwargs.get('out_channels', out_channels)
        self.kernel_size = kwargs.get('kernel_size', 1)
        self.padding = kwargs.get('padding', 0)
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.drop_bl = DropBlock2d(0.2, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.elu(out)
        if out.shape[-1] > 5:
            out = self.drop_bl(out)
        else:
            out = self.dropout(out)
        return out

class CoSOV1(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(CoSOV1, self).__init__()
        self.groups = groups
        self.group_size = in_channels // self.groups
        self.gconvs = nn.ModuleList([
            GConv(self.group_size,
                  self.group_size) for _ in range(groups)
        ])
        self.pconv = PConv(in_channels, out_channels)
    
    def forward(self, x):
        x = torch.reshape(x, (self.groups, self.group_size, *x.shape[2:]))
        out = [conv(x[i].unsqueeze(0)) for i, conv in enumerate(self.gconvs)]
        out = torch.cat(out, dim=1)
        out = self.pconv(out)        
        return out
    
class Residual_CoSOV1(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(Residual_CoSOV1, self).__init__()
        self.cosov1_1 = CoSOV1(in_channels, out_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.cosov1_2 = CoSOV1(out_channels, out_channels, groups)

    def forward(self, x):
        out = self.cosov1_1(x)
        rc = self.conv1(x)
        out = self.cosov1_2(out)
        return torch.add(out, rc)

class EncoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, maxpool_kernel_size=2):
        super(EncoderUnit,self).__init__()
        self.residual_cosov1_1 = Residual_CoSOV1(in_channels, out_channels, groups)
        self.residual_cosov1_2 = Residual_CoSOV1(out_channels, out_channels, groups)
        self.residual_cosov1_3 = Residual_CoSOV1(out_channels, out_channels, groups)
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size, maxpool_kernel_size)

    def forward(self, x):
        out = self.residual_cosov1_1(x)
        out = self.residual_cosov1_2(out)
        transfer = self.residual_cosov1_3(out)
        out = self.maxpool(transfer)
        return out, transfer

class MiddleUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,stride=1,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,1,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.conv1(x)
        rc = self.conv2(x)
        out = self.bn1(out)
        out = self.dropout(out)
        return torch.add(out, rc)

class Dec_Dconv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(in_channels, out_channels, groups // 2, 2, 12,)

    def forward(self, x):
        out = self.dconv(x)
        return out

class Dec_Res_Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.groups = groups
        self.group_size = in_channels // self.groups
        self.gconvs = nn.ModuleList([
            GConv(self.group_size,
                  self.group_size) for _ in range(groups)
        ])
        self.pconv_1 = PConv(in_channels, out_channels)
        self.pconv_2 = PConv(in_channels, out_channels)

    
    def forward(self, x, y):
        out_left = self.pconv_1(y)  
        y = torch.reshape(y, (self.groups, self.group_size, *y.shape[2:]))
        out = [conv(y[i].unsqueeze(0)) for i, conv in enumerate(self.gconvs)]
        out = torch.cat(out, dim=1)
        out = torch.cat((out, x))
        out = self.pconv_2(out)
        return torch.add(out, out_left)

class Dec_Res_Block_old(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU(inplace=True)
        self.drop_bl = DropBlock2d(0.2, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, y):
        out = self.gconv1(y)

        out = self.bn1(out)
        out = self.elu1(out)
        out1 = self.conv1(y)
        out2 = torch.cat((out,x))
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.elu1(out2)
        if out2.shape[-1] > 5:
            out2 = self.drop_bl(out2)
        else:
            out2 = self.dropout(out2)
        return torch.add(out1, out2)

class DecoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groups = out_channels
        self.dec_res_block = Dec_Res_Block(in_channels, out_channels, self.groups)
        self.dec_dconv_block = Dec_Dconv_Block(in_channels, 2, self.groups)

    def forward(self, x, y):
        y_out = self.dec_res_block(x, y)
        out_dl = self.dec_dconv_block(y_out)
        return y_out, out_dl

class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        return x

class CoSOV1Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb2grouped = RGB2ChPairs()
        self.encoder_1 = EncoderUnit(12, 12, 6)
        self.encoder_2 = EncoderUnit(12, 16, 4)
        self.encoder_3 = EncoderUnit(16, 32, 8)
        self.encoder_4 = EncoderUnit(32, 64, 16)
        self.encoder_5 = EncoderUnit(64, 128, 16)
        self.encoder_6 = EncoderUnit(128, 128, 16)
        self.encoder_7 = EncoderUnit(128, 128, 16)
        self.encoder_8 = EncoderUnit(128, 128, 16, maxpool_kernel_size=3)
        self.middle = MiddleUnit(128, 128)
        self.decoder_1 = DecoderUnit(128, 128)
        self.decoder_2 = DecoderUnit(128, 128)
        self.decoder_3 = DecoderUnit(128, 128)
        self.decoder_4 = DecoderUnit(128, 128)
        self.decoder_5 = DecoderUnit(128, 64)
        self.decoder_6 = DecoderUnit(64, 32)
        self.decoder_7 = DecoderUnit(32, 16)
        self.decoder_8 = DecoderUnit(16, 8)
        

    def forward(self, x):
        out = self.rgb2grouped(x)
        out, transfer_1 = self.encoder_1(out)
        out, transfer_2 = self.encoder_2(out)
        out, transfer_3 = self.encoder_3(out)
        out, transfer_4 = self.encoder_4(out)
        out, transfer_5 = self.encoder_5(out)
        out, transfer_6 = self.encoder_6(out)
        out, transfer_7 = self.encoder_7(out)
        out, transfer_8 = self.encoder_8(out)
        out = self.middle(out)
        out = self.decoder_1(transfer_8, out)
        out = self.decoder_2(transfer_7, out)
        out = self.decoder_2(transfer_6, out)
        out = self.decoder_2(transfer_5, out)
        out = self.decoder_2(transfer_4, out)
        out = self.decoder_2(transfer_3, out)
        out = self.decoder_2(transfer_2, out)
        out = self.decoder_2(transfer_1, out)