import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv1d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv1d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class UNet_module(nn.Module):
    def __init__(self, set_in):
        super().__init__()
        self.left_conv_1 = ConvBlock(in_channels=set_in, middle_channels=64, out_channels=64)
        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)

        self.right_conv_1 = ConvBlock(in_channels=64, middle_channels=64, out_channels=64)
        self.right_conv_2 = ConvBlock(in_channels=128, middle_channels=128, out_channels=64)

    def forward(self, x):

        feature_1 = self.left_conv_1(x)
        feature_1_pool, indice1 = self.pool(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool, indice2 = self.pool(feature_2)

        de_feature_2 = self.left_conv_3(feature_2_pool)
        de_feature_2_unpool = self.unpool(de_feature_2, indice2, output_size=feature_2.size())

        temp = feature_2+de_feature_2_unpool
        de_feature_1 = self.right_conv_2(temp)
        de_feature_1_unpool = self.unpool(de_feature_1, indice1, output_size=feature_1.size())

        temp = feature_1 + de_feature_1_unpool
        out = self.right_conv_1(temp)

        return out

class Forward_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.UNet_M1 = UNet_module(64)
        self.UNet_M2 = UNet_module(64)
        self.UNet_M3 = UNet_module(64)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1)
    def forward(self, N, K, d):
        N = N.unsqueeze(1)
        K = K.unsqueeze(1)
        d = d.unsqueeze(1)
        d = d.repeat(1, 1, 288)
        x = torch.cat([N, K, d], dim=1)

        x = self.relu(self.conv(x))
        x = self.UNet_M1(x)
        x = self.UNet_M2(x)
        x = self.UNet_M3(x)
        x = self.conv_out(x)
        x = x.reshape(x.shape[0], -1)
        return x

class Inverse_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.UNet_M1 = UNet_module(64)
        self.UNet_M2 = UNet_module(64)
        self.UNet_M3 = UNet_module(64)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
    def forward(self, x):
        x = x.reshape(x.shape[0], 4, -1)

        x = self.relu(self.conv(x))
        x = self.UNet_M1(x)
        x = self.UNet_M2(x)
        x = self.UNet_M3(x)
        x = self.conv_out(x)
        N, K, t = x.split([1, 1, 1], dim=1)
        t = torch.mean(t, dim=2, keepdim=True)
        return N.squeeze(1), K.squeeze(1), t.squeeze(1)
