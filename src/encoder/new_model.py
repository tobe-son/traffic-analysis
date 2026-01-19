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


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        width = out_channels
        self.conv1 = torch.nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(width)
        self.conv2 = torch.nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(width)
        self.conv3 = torch.nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)

        out_channels_expanded = out_channels * self.expansion
        if stride != 1 or in_channels != out_channels_expanded:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels_expanded, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels_expanded),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

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


class _Encoder_ResNetBase(torch.nn.Module):
    def __init__(
        self,
        block: type[torch.nn.Module],
        layers: list[int],
        stem_channels: int = 64,
        out_channels: int = 32,
    ):
        super().__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(1, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(stem_channels),
            torch.nn.ReLU(inplace=True),
        )

        self.inplanes = stem_channels
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        final_in = 512 * getattr(block, "expansion", 1)
        self.final_conv = torch.nn.Conv2d(final_in, out_channels, kernel_size=1)

    def _make_layer(self, block: type[torch.nn.Module], planes: int, blocks: int, stride: int) -> torch.nn.Sequential:
        layers = [block(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * getattr(block, "expansion", 1)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)
        return x


class Encoder_ResNet18(_Encoder_ResNetBase):
    def __init__(self):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])


class Encoder_ResNet50(_Encoder_ResNetBase):
    def __init__(self):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])

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