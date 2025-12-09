import torch
import torch.nn.functional as F

"""
CNN_s
"""

class Encoder_Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        # print(f"Encoder Input image size: {x.size()}\n")
        x = self.encoder(x)
        # print(f"Encoder Output image size: {x.size()}\n")
        return x

class Decoder_Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            # torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Decoder Input image size: {x.size()}\n")
        x = self.decoder(x)
        # print(f"Decoder Output image size: {x.size()}\n")
        return x

class AutoEncoder_Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder_Small()
        self.dec = Decoder_Small()
        
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
    
"""
CNN_o
"""

class Encoder_Original(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        # print(f"Encoder Input image size: {x.size()}\n")
        x = self.encoder(x)
        # print(f"Encoder Output image size: {x.size()}\n")
        return x

class Decoder_Original(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Decoder Input image size: {x.size()}\n")
        x = self.decoder(x)
        # print(f"Decoder Output image size: {x.size()}\n")
        return x

class AutoEncoder_Original(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder_Original()
        self.dec = Decoder_Original()
        
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

"""
Wave1D（1次元畳み込みオートエンコーダ）
"""


class Encoder_Wave1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_Wave1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder_Wave1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder_Wave1D()
        self.dec = Decoder_Wave1D()

    def forward(self, x):
        latent = self.enc(x)
        return self.dec(latent)