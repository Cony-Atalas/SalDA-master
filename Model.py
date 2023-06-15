import torch
import torch.nn as nn
from attention.SA_Block import SA_Block
from attention.SC_Block import SC1_Block,SC2_Block,SC3_Block


class SalDA(nn.Module):
    def __init__(self, dataset = 'SALICON'):
        super(SalDA,self).__init__()
        self.dataset = dataset
        #loading pretrained weights in dictionary './Models'
        self.d = torch.load('./Models/vggm-786f2434.pth')

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 7,stride=1,padding=3)
        self.act1 = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(5)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride=1, padding =1)
        self.act3 = nn.ReLU()
        self.sc3 = SC3_Block(in_channel=512)

        self.conv4 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 5, stride=1, padding =2)
        self.act4 = nn.ReLU()
        self.sc4 = SC2_Block(in_channels=512)

        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 5, stride=1, padding =2)
        self.act5 = nn.ReLU()
        self.sc5 = SC3_Block(in_channel=512)
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 7, stride=1, padding =3)
        self.act6 = nn.ReLU()
        self.sc6 = SC2_Block(in_channels=256)

        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 11, stride=1, padding =5)
        self.act7 = nn.ReLU()
        self.sc7 = SC3_Block(in_channel=128)
        self.conv8 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 11, stride=1, padding =5)
        self.act8 = nn.ReLU()
        self.sc8 = SC2_Block(in_channels=32)
        # self.filter

        self.conv9 = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 13, stride=1, padding =6)
        self.sa = SA_Block(in_channels=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels= 1 , out_channels= 1 ,
                                          kernel_size=8 , stride=4 , padding = 2,bias = False)



    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.lrn(x)
        x = self.maxpool1(x)

        x = self.act2(self.conv2(x))
        x = self.maxpool2(x)

        x = self.act3(self.conv3(x))
        x = self.sc3(x)

        x = self.act4(self.conv4(x))
        x = self.sc4(x)
        x = self.act5(self.conv5(x))
        x = self.sc5(x)
        x = self.act6(self.conv6(x))
        x =self.sc6(x)
        x = self.act7(self.conv7(x))
        x = self.sc7(x)
        x = self.act8(self.conv8(x))
        x = self.sc8(x)
        x = self.conv9(x)
        x = self.sa(x)

        x = self.deconv1(x)
        return x
    
if __name__ == '__main__':
    model = SalDA().cuda()
    input = torch.randn(4, 3, 240, 320).cuda()
    output = model(input)
    print(output.size())
