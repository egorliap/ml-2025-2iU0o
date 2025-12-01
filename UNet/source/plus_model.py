import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNetPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) implementation.
    - in_channels: входные каналы (обычно 3)
    - out_channels: число классов / выходных каналов (например, 3 для Oxford-IIIT Pet segmentation)
    - init_features: количество фичей на уровне 0 (будет удваиваться вниз по энкодеру)
    - depth: глубина (по умолчанию 4: уровни 0..4)
    - deep_supervision: если True, модель вернёт список промежуточных предсказаний (conv0_1..conv0_depth)
      и тогда в тренинге нужно суммировать/взвесить loss для каждого.
    """
    def __init__(self, in_channels=3, out_channels=3, init_features=32, depth=4, deep_supervision=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.depth = depth
        self.deep_supervision = deep_supervision

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.feat_channels = []
        for i in range(depth + 1):
            out_ch = init_features * (2 ** i)
            in_ch = in_channels if i == 0 else init_features * (2 ** (i - 1))
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch))
            self.feat_channels.append(out_ch)

        self.pool = nn.MaxPool2d(2, 2)

        # --- Nested decoder connections ---
        self.nested_convs = nn.ModuleDict()
        for j in range(1, depth + 1):
            for i in range(0, depth - j + 1):
                feat_i = init_features * (2 ** i)
                feat_ip1 = init_features * (2 ** (i + 1))
                in_ch = feat_i * j + feat_ip1
                out_ch = feat_i
                self.nested_convs[f"conv{i}_{j}"] = ConvBlock(in_ch, out_ch)

        self.final_conv = nn.Conv2d(init_features, out_channels, 1)

    def forward(self, x):
        nodes = {}

        # Encoder path
        current = x
        for i in range(self.depth + 1):
            nodes[f"conv{i}_0"] = self.encoder_blocks[i](current)
            if i != self.depth:
                current = self.pool(nodes[f"conv{i}_0"])

        # Decoder path (nested)
        for j in range(1, self.depth + 1):
            for i in range(0, self.depth - j + 1):
                inputs = [nodes[f"conv{i}_{k}"] for k in range(j)]
                up = F.interpolate(nodes[f"conv{i+1}_{j-1}"], size=inputs[0].shape[2:], mode='bilinear', align_corners=True)
                inputs.append(up)
                cat = torch.cat(inputs, dim=1)
                nodes[f"conv{i}_{j}"] = self.nested_convs[f"conv{i}_{j}"](cat)

        # Выход: только последний conv0_depth
        out = self.final_conv(nodes[f"conv0_{self.depth}"])
        return out
