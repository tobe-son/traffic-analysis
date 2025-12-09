import torch
import torch.nn.functional as F

"""
CNN_VGG11
"""

class Encoder_VGG11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.block3_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = torch.nn.Conv2d(256, 32, kernel_size=1)
        self._pool_shapes = None

    def forward(self, x):
        self._pool_shapes = []
        x = self.initial(x)
        x = self.block1(x)
        self._pool_shapes.append(x.shape[-2:])
        x = self.pool1(x)

        x = self.block2(x)
        self._pool_shapes.append(x.shape[-2:])
        x = self.pool2(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        self._pool_shapes.append(x.shape[-2:])
        x = self.pool3(x)

        x = self.final_conv(x)
        return x

    @property
    def pool_shapes(self):
        if self._pool_shapes is None:
            raise ValueError("Encoder_VGG11 forward must run before accessing pool_shapes.")
        return tuple(self._pool_shapes)

class Decoder_VGG11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(32, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.reconstruction = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, pool_shapes=None):
        if pool_shapes is None or len(pool_shapes) != 3:
            raise ValueError("Decoder_VGG11 expects three encoder pool shapes.")
        target1, target2, target3 = pool_shapes

        x = self.expand(x)
        x = F.interpolate(x, size=target3, mode='bilinear', align_corners=False)
        x = self.block3(x)

        x = F.interpolate(x, size=target2, mode='bilinear', align_corners=False)
        x = self.block2(x)

        x = F.interpolate(x, size=target1, mode='bilinear', align_corners=False)
        x = self.block1(x)
        x = self.reconstruction(x)
        return x

class AutoEncoder_VGG11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder_VGG11()
        self.dec = Decoder_VGG11()

    def forward(self, x):
        latent = self.enc(x)
        x = self.dec(latent, self.enc.pool_shapes)
        return x

"""
CNN_resnet
"""

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

class Encoder_ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.final_conv = torch.nn.Conv2d(256, 32, kernel_size=1)
        self._feature_shapes = None

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        shapes = []
        x = self.stem(x)
        shapes.append(x.shape[-2:])
        x = self.layer1(x)
        shapes.append(x.shape[-2:])
        x = self.layer2(x)
        shapes.append(x.shape[-2:])
        x = self.layer3(x)
        shapes.append(x.shape[-2:])
        x = self.final_conv(x)
        self._feature_shapes = tuple(shapes)
        return x

    @property
    def feature_shapes(self):
        if self._feature_shapes is None:
            raise ValueError("Encoder_ResNet forward must run before accessing feature_shapes.")
        return self._feature_shapes

class Decoder_ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(32, 256, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.block3 = ResidualBlock(256, 256)
        self.block2_reduce = ResidualBlock(256, 128)
        self.block2 = ResidualBlock(128, 128)
        self.block1_reduce = ResidualBlock(128, 64)
        self.block1 = ResidualBlock(64, 64)
        self.reconstruction = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, feature_shapes=None):
        if feature_shapes is None or len(feature_shapes) != 4:
            raise ValueError('Decoder_ResNet expects four encoder feature shapes.')
        shape0, shape1, shape2, shape3 = feature_shapes

        x = self.expand(x)
        x = self.block3(x)
        x = F.interpolate(x, size=shape3, mode='bilinear', align_corners=False)
        x = self.block2_reduce(x)
        x = self.block2(x)
        x = F.interpolate(x, size=shape2, mode='bilinear', align_corners=False)
        x = self.block1_reduce(x)
        x = self.block1(x)
        x = F.interpolate(x, size=shape1, mode='bilinear', align_corners=False)
        x = self.reconstruction(x)
        x = F.interpolate(x, size=shape0, mode='bilinear', align_corners=False)
        return x

class AutoEncoder_ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder_ResNet()
        self.dec = Decoder_ResNet()

    def forward(self, x):
        latent = self.enc(x)
        x = self.dec(latent, self.enc.feature_shapes)
        return x