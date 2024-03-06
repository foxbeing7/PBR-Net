import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc(avg_pool.view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        max_out = self.fc(max_pool.view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        attention = torch.sigmoid(avg_out + max_out)
        return attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return torch.sigmoid(attention)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        out = x * channel_attention * spatial_attention
        return out
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            CBAM(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            CBAM(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




class normal_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(normal_UNet, self).__init__()
        self.filters = 16
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=self.filters*2)
        self.Conv2 = conv_block(ch_in=self.filters*2, ch_out=self.filters*4)
        self.Conv3 = conv_block(ch_in=self.filters*4, ch_out=self.filters*8)
        self.Conv4 = conv_block(ch_in=self.filters*8, ch_out=self.filters*16)
        self.Conv5 = conv_block(ch_in=self.filters*16, ch_out=self.filters*32)

        self.Up5 = up_conv(ch_in=self.filters*32, ch_out=self.filters*16)
        self.Up_conv5 = conv_block(ch_in=self.filters*32, ch_out=self.filters*16)

        self.Up4 = up_conv(ch_in=self.filters*16, ch_out=self.filters*8)
        self.Up_conv4 = conv_block(ch_in=self.filters*16, ch_out=self.filters*8)

        self.Up3 = up_conv(ch_in=self.filters*8, ch_out=self.filters*4)
        self.Up_conv3 = conv_block(ch_in=self.filters*8, ch_out=self.filters*4)

        self.Up2 = up_conv(ch_in=self.filters*4, ch_out=self.filters*2)
        self.Up_conv2 = conv_block(ch_in=self.filters*4, ch_out=self.filters*2)

        self.Conv_1x1 = nn.Conv2d(self.filters*2, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # print("normal net output: {}", d1)
        return d1

class albedo_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(albedo_UNet, self).__init__()

        self.filters = 16
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=self.filters*2)
        self.Conv2 = conv_block(ch_in=self.filters*2, ch_out=self.filters*4)
        self.Conv3 = conv_block(ch_in=self.filters*4, ch_out=self.filters*8)
        self.Conv4 = conv_block(ch_in=self.filters*8, ch_out=self.filters*16)
        self.Conv5 = conv_block(ch_in=self.filters*16, ch_out=self.filters*32)

        self.Up5 = up_conv(ch_in=self.filters*32, ch_out=self.filters*16)
        self.Up_conv5 = conv_block(ch_in=self.filters*32, ch_out=self.filters*16)

        self.Up4 = up_conv(ch_in=self.filters*16, ch_out=self.filters*8)
        self.Up_conv4 = conv_block(ch_in=self.filters*16, ch_out=self.filters*8)

        self.Up3 = up_conv(ch_in=self.filters*8, ch_out=self.filters*4)
        self.Up_conv3 = conv_block(ch_in=self.filters*8, ch_out=self.filters*4)

        self.Up2 = up_conv(ch_in=self.filters*4, ch_out=self.filters*2)
        self.Up_conv2 = conv_block(ch_in=self.filters*4, ch_out=self.filters*2)

        self.Conv_1x1 = nn.Conv2d(self.filters*2, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # print("albedo net output: {}",d1)
        return d1
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.filters = 16
        self.net = nn.Sequential(
            nn.Conv2d(3, self.filters*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*2, self.filters*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.filters*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*2, self.filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.filters*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*4, self.filters*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.filters*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*4, self.filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.filters*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*8, self.filters*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.filters*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*8, self.filters*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.filters*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.filters*16, self.filters*16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.filters*16),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.filters*16, self.filters*32, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.filters*32, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))

class pbr_unet(nn.Module):
    def __init__(self,normal_UNet,albedo_UNet):
        super(pbr_unet,self).__init__()
        self.normal_UNet = normal_UNet
        self.albedo_UNet = albedo_UNet

    def forward(self,x):

        normal_output = self.normal_UNet(x)
        albedo_output = self.albedo_UNet(x)
        # albedo_output = self.albedo_UNet(torch.cat((res,normal_output),dim=1))

        return normal_output,albedo_output
