import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv1d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv1d(in_channels=middle_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv1d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class ReflectoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_conv_1 = ConvBlock(in_channels=4, middle_channels=64, out_channels=64)
        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.middle_conv = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)
        self.right_conv_1 = ConvBlock(in_channels=128, middle_channels=64, out_channels=3)
        self.right_conv_2 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)
        self.right_conv_3 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)
        self.right_conv_4 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)

        self.up_conv_4 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, stride=2, kernel_size=2)
        self.up_conv_3 = nn.ConvTranspose1d(in_channels=512, out_channels=512, stride=2, kernel_size=2)
        self.up_conv_2 = nn.ConvTranspose1d(in_channels=256, out_channels=256, stride=2, kernel_size=2)
        self.up_conv_1 = nn.ConvTranspose1d(in_channels=128, out_channels=128, stride=2, kernel_size=2)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 4, -1)

        x = self.relu(self.left_conv_1(x))
        x = self.pool(x)

        x = self.relu(self.left_conv_2(x))
        x = self.pool(x)

        x = self.relu(self.left_conv_3(x))
        x = self.pool(x)

        x = self.relu(self.left_conv_4(x))
        x = self.pool(x)

        x = self.relu(self.middle_conv(x))
        x = self.up_conv_4(x)

        x = self.relu(self.right_conv_4(x))
        x = self.up_conv_3(x)

        x = self.relu(self.right_conv_3(x))
        x = self.up_conv_2(x)

        x = self.relu(self.right_conv_2(x))
        x = self.up_conv_1(x)

        x = self.right_conv_1(x)

        N, K, t = x.split([1, 1, 1], dim=1)
        t = torch.mean(t, dim=2, keepdim=True)
        return N.squeeze(1), K.squeeze(1), t.squeeze(1)
