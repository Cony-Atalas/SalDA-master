import torch
from torch import nn

# channel attention module for SalDA
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.batchnorm = nn.BatchNorm1d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = (self.max_pool(x)+self.avg_pool(x)).view(b, c)
        y = self.batchnorm(y)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == '__main__':
    model = CA_Block(channel=16).cuda()
    input = torch.randn(4,16,240,320).cuda()
    output = model(input)
    print(output.size())
