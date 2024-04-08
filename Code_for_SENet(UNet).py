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

class SENet_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.UNet_M = UNet_module(64)
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

        self.fn1_b = nn.Linear(288, 100)
        self.fn2_b = nn.Linear(100, 100)
        self.fn1_pole = nn.Linear(288, 100)
        self.fn2_pole = nn.Linear(100, 100)
        self.fn1_t = nn.Linear(288, 100)
        self.fn2_t = nn.Linear(100, 100)

        self.fn_b0 = nn.Linear(100, 1)
        self.fn_b1 = nn.Linear(100, 42)
        self.fn_b2 = nn.Linear(100, 2)

        self.fn_Einf = nn.Linear(100, 1)
        self.fn_UA = nn.Linear(100, 1)

        self.fn_t = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(x.shape[0], 4, -1)

        x = self.relu(self.conv(x))
        x = self.UNet_M(x)
        x = self.conv_out(x).squeeze(1)

        x_b = self.relu(self.fn1_b(x))
        x_b = self.relu(self.fn2_b(x_b))
        b0 = self.fn_b0(x_b)
        b1 = self.softplus(self.fn_b1(x_b))
        b2 = self.fn_b2(x_b)
        b = torch.cat((b0, b1, b2), 1)

        x_pole = self.relu(self.fn1_pole(x))
        x_pole = self.relu(self.fn2_pole(x_pole))
        Einf = self.fn_Einf(x_pole)
        UA = self.fn_UA(x_pole)
        p = torch.cat((Einf, UA), 1)

        x_t = self.relu(self.fn1_t(x))
        x_t = self.relu(self.fn2_t(x_t))
        t = self.fn_t(x_t)

        return b, p, t

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.UNet_M = UNet_module(64)
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], 4, -1)

        x = self.relu(self.conv(x))
        x = self.UNet_M(x)
        x = self.conv_out(x).squeeze(1)

        return x

class BranchNet_BsplineCoefficients(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn1_b = nn.Linear(288, 100)
        self.fn2_b = nn.Linear(100, 100)

        self.fn_b0 = nn.Linear(100, 1)
        self.fn_b1 = nn.Linear(100, 42)
        self.fn_b2 = nn.Linear(100, 2)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        pass

    def forward(self, x):
        x_b = self.relu(self.fn1_b(x))
        x_b = self.relu(self.fn2_b(x_b))

        b0 = self.fn_b0(x_b)
        b1 = self.softplus(self.fn_b1(x_b))
        b2 = self.fn_b2(x_b)

        b = torch.cat((b0, b1, b2), 1)

        return b
        pass

class BranchNet_pole(nn.Module):
    def __init__(self):
        super().__init__()

        self.fn1_pole = nn.Linear(288, 100)
        self.fn2_pole = nn.Linear(100, 100)

        self.fn_Einf = nn.Linear(100, 1)
        self.fn_UA = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        pass

    def forward(self, x):

        x_pole = self.relu(self.fn1_pole(x))
        x_pole = self.relu(self.fn2_pole(x_pole))
        Einf = self.fn_Einf(x_pole)
        UA = self.fn_UA(x_pole)
        p = torch.cat((Einf, UA), 1)

        return p
        pass

class BranchNet_t(nn.Module):
    def __init__(self):
        super().__init__()

        self.fn1_t = nn.Linear(288, 100)
        self.fn2_t = nn.Linear(100, 100)

        self.fn_t = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        pass

    def forward(self, x):
        x_t = self.relu(self.fn1_t(x))
        x_t = self.relu(self.fn2_t(x_t))
        t = self.tanh(self.fn_t(x_t))

        return t
        pass
