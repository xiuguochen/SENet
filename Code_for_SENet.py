import torch
from torch import nn

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            pass
        pass
    pass

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, dowmsampling=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if dowmsampling:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
        else:
            self.conv3 = None
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)
    pass

class SENet_train(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=5, stride=1, padding=2),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=2, stride=2))

        b2 = nn.Sequential(*self.__resnet_block(64, 64, 3, Fisrtblock=True))
        b3 = nn.Sequential(*self.__resnet_block(64, 96, 4))
        b4 = nn.Sequential(*self.__resnet_block(96, 128, 6))
        b5 = nn.Sequential(*self.__resnet_block(128, 128, 3))
        self.conv = nn.Sequential(b1, b2, b3, b4, b5, nn.AvgPool1d(2), nn.Flatten())

        self.fn1_b = nn.Linear(9 * 128, 100)
        self.fn2_b = nn.Linear(100, 100)
        self.fn1_pole = nn.Linear(9 * 128, 100)
        self.fn2_pole = nn.Linear(100, 100)
        self.fn1_t = nn.Linear(9 * 128, 100)
        self.fn2_t = nn.Linear(100, 100)

        self.fn_b0 = nn.Linear(100, 1)
        self.fn_b1 = nn.Linear(100, 42)
        self.fn_b2 = nn.Linear(100, 2)

        self.fn_Einf = nn.Linear(100, 1)
        self.fn_UA = nn.Linear(100, 1)

        self.fn_t = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        pass

    def forward(self, x):
        x = x.reshape(x.shape[0], 4, -1)
        x = self.conv(x)

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
        pass

    def __resnet_block(self, input_channels, num_channels, num_residuals, Fisrtblock=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not Fisrtblock:
                blk.append(Residual(input_channels, num_channels, dowmsampling=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=5, stride=1, padding=2),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=2, stride=2))

        b2 = nn.Sequential(*self.__resnet_block(64, 64, 3, Fisrtblock=True))
        b3 = nn.Sequential(*self.__resnet_block(64, 96, 4))
        b4 = nn.Sequential(*self.__resnet_block(96, 128, 6))
        b5 = nn.Sequential(*self.__resnet_block(128, 128, 3))
        self.conv = nn.Sequential(b1, b2, b3, b4, b5, nn.AvgPool1d(2), nn.Flatten())
        pass

    def forward(self, x):
        x = x.reshape(x.shape[0], 4, -1)
        x = self.conv(x)
        return x
        pass

    def __resnet_block(self, input_channels, num_channels, num_residuals, Fisrtblock=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not Fisrtblock:
                blk.append(Residual(input_channels, num_channels, dowmsampling=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

class BranchNet_BsplineCoefficients(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn1_b = nn.Linear(9 * 128, 100)
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

        self.fn1_pole = nn.Linear(9 * 128, 100)
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

        self.fn1_t = nn.Linear(9 * 128, 100)
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

class AuxiliaryNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fn1_g = nn.Linear(9 * 128, 100)
        self.fn2_g = nn.Linear(100, 100)

        self.fn_g = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        pass

    def forward(self, x):
        x_g = self.relu(self.fn1_g(x))
        x_g = self.relu(self.fn2_g(x_g))
        g = self.tanh(self.fn_g(x_g))

        return g
        pass

