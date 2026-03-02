# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:03:41 2026

@author: kubota
"""

# model/condnet_tart.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
#  Building blocks
# ----------------------------
class TwoConvBlock(nn.Module):  # TCB
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, C) -> MultiheadAttention expects (N, B, C) unless batch_first
        # In your original code you used attention(x,x,x) with x as (B,N,C),
        # but default MultiheadAttention is NOT batch_first. We'll keep your behavior
        # by permuting inside.
        x_t = x.permute(1, 0, 2)  # (N,B,C)
        attn = self.attention(x_t, x_t, x_t)[0]  # (N,B,C)
        attn = attn.permute(1, 0, 2)  # (B,N,C)

        x = self.dropout(self.norm1(attn + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x


# ----------------------------
#  UNet backbone (produces intermediate feature maps)
# ----------------------------
class UNet_2D(nn.Module):
    """
    Input: (img1, img2) each [B,1,H,W]
    Output: dict {t6,t7,t8,t9,t}  (same as your original)
    """
    def __init__(self, num_classes=10, transformer_count=6):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 32, 32)
        self.TCB2 = TwoConvBlock(32, 64, 64)
        self.TCB3 = TwoConvBlock(64, 128, 128)
        self.TCB4 = TwoConvBlock(128, 256, 256)
        self.TCB5 = TwoConvBlock(256, 512, 512)
        self.TCB6 = TwoConvBlock(512, 256, 256)
        self.TCB7 = TwoConvBlock(256, 128, 128)
        self.TCB8 = TwoConvBlock(128, 64, 64)
        self.TCB9 = TwoConvBlock(64, 32, 32)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.UC1 = UpConv(512, 256)
        self.UC2 = UpConv(256, 128)
        self.UC3 = UpConv(128, 64)
        self.UC4 = UpConv(64, 32)

        # NOTE: in your original, conv1 exists but its output isn't used downstream.
        # We keep it for compatibility.
        self.conv1 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

        self.Trans = nn.ModuleList(
            [TransformerBlock(embed_size=512, heads=4) for _ in range(transformer_count)]
        )

    def forward(self, a, b, mask):
        x = torch.cat([a, b, mask], dim=1)  # [B,2,H,W]
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)  # [B,512,H/16,W/16]

        # Transformer (flatten spatial)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C) where N=H*W

        for transformer in self.Trans:
            x = transformer(x)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # back to [B,512,H,W]
        t = x

        x = self.UC1(x)
        x = torch.cat([x4, x], dim=1)
        x = self.TCB6(x)
        t6 = x

        x = self.UC2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.TCB7(x)
        t7 = x

        x = self.UC3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.TCB8(x)
        t8 = x

        x = self.UC4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.TCB9(x)
        t9 = x

        _ = self.conv1(x)  # kept for compatibility, not used

        return {"t6": t6, "t7": t7, "t8": t8, "t9": t9, "t": t}


# ----------------------------
#  Transfer head (conductivity regression)
# ----------------------------
class MapMod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.r = nn.ReLU()

        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.identity_conv = None

    def forward(self, x):
        identity = self.identity_conv(x) if self.identity_conv else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.r(x)
        x = self.conv2(x)
        x = x + identity
        x = self.r(x)
        return x


class UpConvwithResConnectforhMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.s = nn.Sigmoid()

        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.identity_conv = None

    def forward(self, x):
        x = self.up(x)
        identity = self.identity_conv(x) if self.identity_conv else x
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = x + identity
        x = self.s(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gating_channels, in_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        alpha = torch.sigmoid(psi)
        return x * alpha


class CondNet_Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.UCRCM1 = UpConvwithResConnectforhMap(512, 256)
        self.UCRCM2 = UpConvwithResConnectforhMap(512, 128)
        self.UCRCM3 = UpConvwithResConnectforhMap(256, 64)
        self.UCRCM4 = UpConvwithResConnectforhMap(128, 32)

        self.att1 = AttentionGate(256, 256)
        self.att2 = AttentionGate(128, 128)
        self.att3 = AttentionGate(64, 64)
        self.att4 = AttentionGate(32, 32)

        self.MAP = MapMod(64, 1, 5)

    def forward(self, outputs):
        t6 = outputs["t6"]
        t7 = outputs["t7"]
        t8 = outputs["t8"]
        t9 = outputs["t9"]
        t  = outputs["t"]

        x = self.UCRCM1(t)
        x1 = x
        x = self.att1(t6, x)
        x = torch.cat([x1, x], dim=1)

        x = self.UCRCM2(x)
        x2 = x
        x = self.att2(t7, x)
        x = torch.cat([x2, x], dim=1)

        x = self.UCRCM3(x)
        x3 = x
        x = self.att3(t8, x)
        x = torch.cat([x3, x], dim=1)

        x = self.UCRCM4(x)
        x4 = x
        x = self.att4(t9, x)
        x = torch.cat([x4, x], dim=1)

        x = self.MAP(x)  # [B,1,H,W]
        return x


# ----------------------------
#  CondNet-TART wrapper (one "model" file, one "model" concept)
# ----------------------------
class CondNetTART(nn.Module):
    """
    One bundled model:
      - backbone: UNet_2D
      - head: CondNet_Transfer
    Forward returns conductivity map [B,1,H,W].

    Usage:
      model = CondNetTART(num_classes=10, transformer_count=6)
      pred = model(img1, img2)
    """
    def __init__(self, num_classes=10, transformer_count=6):
        super().__init__()
        self.backbone = UNet_2D(num_classes=num_classes, transformer_count=transformer_count)
        self.head = CondNet_Transfer()

    def forward(self, img1, img2):
        feats = self.backbone(img1, img2)
        out = self.head(feats)
        return out

    def freeze_backbone_parts(self, keys=("TCB1","TCB2","TCB3","TCB4","TCB5","Trans")):
        """
        Freeze by name pattern (same logic as your original).
        Call AFTER loading weights if you want.
        """
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in keys):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def set_frozen_transformer_eval(self):
        """
        In training, you sometimes want frozen Transformer blocks to stay in eval mode.
        Call each epoch (or once) as you did.
        """
        for blk in self.backbone.Trans:
            blk.eval()