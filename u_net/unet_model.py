import torch
import torch.nn as nn
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"


# 定义卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(),
                                  )

    def forward(self, input):
        out = self.conv(input)
        return out


# 定义Unet网络
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        """定义下采样网络"""
        self.encoder1 = DoubleConv(1, 32)
        self.encoder1_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder2_down = nn.MaxPool2d(2, 2)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder3_down = nn.MaxPool2d(2, 2)
        self.encoder4 = DoubleConv(128, 256)
        self.encoder4_down = nn.MaxPool2d(2, 2)
        self.encoder5 = DoubleConv(256, 512)

        """定义上采样网络"""
        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.decoder1_up = DoubleConv(512, 256)

        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2))
        self.decoder2_up = DoubleConv(256, 128)

        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2))
        self.decoder3_up = DoubleConv(128, 64)

        self.decoder4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2))
        self.decoder4_up = DoubleConv(64, 32)

        self.decoder_output = nn.Conv2d(32, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e1_down = self.encoder1_down(e1)
        e2 = self.encoder2(e1_down)
        e2_down = self.encoder2_down(e2)
        e3 = self.encoder3(e2_down)
        e3_down = self.encoder3_down(e3)
        e4 = self.encoder4(e3_down)
        e4_down = self.encoder4_down(e4)
        e5 = self.encoder5(e4_down)

        d1 = self.decoder1(e5)
        d1 = torch.cat((d1, e4), dim=1)
        d1 = self.decoder1_up(d1)

        d2 = self.decoder2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.decoder2_up(d2)

        d3 = self.decoder3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3_up(d3)

        d4 = self.decoder4(d3)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.decoder4_up(d4)

        out = self.decoder_output(d4)
        # print(out.shape)

        return out


if __name__ == '__main__':
    summary(Unet().to(device), input_size=(1, 320, 480), batch_size=-1)
