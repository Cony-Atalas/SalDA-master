import torch
from torch import nn
from torch.nn import functional as F


# spatial attention module for SalDA
class SA_Block(nn.Module):
    def __init__(self,in_channels):
        super(SA_Block, self).__init__()

        self.in_channels = in_channels
        # self.inter_channels = inter_channels

        # if self.inter_channels is None:
        #     self.inter_channels = in_channels // 2
        #     if self.inter_channels == 0:
        #         self.inter_channels = 1

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,
                               kernel_size=1,stride=1,padding=0)

        self.conv2 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,
                               kernel_size=1,stride=1,padding=0)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=1,
                               kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(1))
        self.maxpool = nn.MaxPool2d(kernel_size=(1,in_channels),stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        batch_size = x.size(0)
        H = x.size(2)
        W = x.size(3)
        conv3_x = self.conv3(x)
        conv3_x = conv3_x.view(batch_size,1,-1)
        conv3_x = conv3_x.permute(0,2,1)

        conv1_x = self.conv1(x).view(batch_size,self.in_channels,-1)
        conv1_x = conv1_x.permute(0,2,1)
        conv2_x = self.conv2(x).view(batch_size,self.in_channels,-1)

        mul_1 = torch.matmul(conv1_x,conv2_x)

        conv3_x = F.softmax(conv3_x,-1)
        # conv3_x = conv3_x.expand(batch_size,conv3_x.size(1),self.in_channels)

        mul_2 = torch.matmul(mul_1,conv3_x)
        # w = self.maxpool(mul_2)
        w = self.sigmoid(mul_2)
        w = w.permute(0,2,1)
        w = w.view(batch_size,1,H,W)
        edit_x = torch.mul(x,w)

        return edit_x+x

# if __name__ == '__main__':
#     torch.device('cuda')
#     model = Edit_NLBlock(in_channels=4)
#     input = torch.Tensor(4, 4, 24, 32)
#     output = model(input)
#     print(output.size())

