import torch

class RGB2ChPairs():
    def __init__(self, chuncks=3):
        self.chuncks = chuncks

    def __call__(self, x):
        r, g, b = torch.chunk(x, self.chuncks, dim=1)
        rr = torch.cat([r, r], dim=1)
        rg = torch.cat([r, g], dim=1) 
        rb = torch.cat([r, b], dim=1)
        gg = torch.cat([g, g], dim=1)
        gb = torch.cat([g, b], dim=1)
        bb = torch.cat([b, b], dim=1)
        out = torch.cat([rr, rg, rb, gg, gb, bb], dim=1)
        return out