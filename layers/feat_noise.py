import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std
        self.buffer = None

    def forward(self, x):
        if self.std > 0:
            if self.buffer is None:
                self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            else:
                self.buffer.data.resize_(x.size()).normal_(0, self.std)
            return x + self.buffer
        return x
