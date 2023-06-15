from torch import nn
from .CA_Block import CA_Block
from .SA_Block import SA_Block
# three different dual attention modules for SalDA

# attention module arrangement (b)
class SC1_Block(nn.Module):

    def __init__(self,in_channels):
        super(SC1_Block,self).__init__()
        self.ca=CA_Block(channel=in_channels)
        self.sa=SA_Block(in_channels=in_channels)

    def forward(self,x):
        b,c,_,_ = x.size()
        # residual = x
        out = x+self.sa(x)
        result = self.ca(out)
        return result+out


# attention module arrangement (a)
class SC2_Block(nn.Module):

    def __init__(self,in_channels):
        super(SC2_Block,self).__init__()
        self.ca=CA_Block(channel=in_channels)
        self.sa=SA_Block(in_channels=in_channels)

    def forward(self,x):
        residual = x
        out = self.sa(x)
        out = self.ca(out)
        return out+residual

# attention module arrangement (c)
class SC3_Block(nn.Module):
    def __init__(self,in_channel,norm_layer=nn.BatchNorm2d):
        super(SC3_Block, self).__init__()

        inter_channel = in_channel // 4

        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channel,inter_channel,3,padding=1,bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channel,inter_channel,3,padding=1,bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )

        self.sa = SA_Block(inter_channel)
        self.ca = CA_Block(inter_channel)

        self.conv_21 = nn.Sequential(
            nn.Conv2d(inter_channel,inter_channel,3,padding=1,bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )
        self.conv_22 = nn.Sequential(
            nn.Conv2d(inter_channel,inter_channel,3,padding=1,bias=False),
            norm_layer(inter_channel),
            nn.ReLU()
        )

        # self.conv3 = nn.Sequential(nn.Dropout2d(0.1,False),nn.Conv2d(inter_channel,in_channel,1))
        # self.conv4 = nn.Sequential(nn.Dropout2d(0.1,False),nn.Conv2d(inter_channel,in_channel,1))

        # self.conv5 = nn.Sequential(nn.Dropout2d(0.1,False),nn.Conv2d(inter_channel,in_channel,1))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channel,in_channel,1))

    def forward(self,x):
        x1 = self.conv_a(x)
        x1 = self.sa(x1)
        x1_conv = self.conv_21(x1)
        # x1_sa_output = self.conv3(x1_conv)

        x2 = self.conv_b(x)
        x2 = self.ca(x2)
        x2_conv = self.conv_22(x2)
        # x2_ca_output = self.conv4(x2_conv)

        feature_sum = x1_conv+x2_conv
        saca_output = self.conv5(feature_sum)

        return saca_output
