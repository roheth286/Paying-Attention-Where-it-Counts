import torch
import torch.nn as nn
from .unet import Encoder, ConvBlock

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # Decoder feature transformation
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        # Encoder skip transformation
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        # Attention coefficient computation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x = encoder skip connection
        # g = decoder gating signal
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = x * psi
        return out


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention_intermediate_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.attention = AttentionGate(
            F_g=out_channels,
            F_l=out_channels,
            F_int=attention_intermediate_channels
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        # Upsample decoder feature
        x = self.upconv(x)

        # Filter encoder skip
        skip = self.attention(skip, x)

        # Concatenate
        x = torch.cat([skip, x], dim=1)

        # Convolution block
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder + Attention
        self.dec1 = AttentionDecoderBlock(1024, 512, 256)
        self.dec2 = AttentionDecoderBlock(512, 256, 128)
        self.dec3 = AttentionDecoderBlock(256, 128, 64)
        self.dec4 = AttentionDecoderBlock(128, 64, 32)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        x = self.final(x)
        return x
