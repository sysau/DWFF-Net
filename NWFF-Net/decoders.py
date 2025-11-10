import torch
import torch.nn as nn
import torch.nn.functional as F

def make_group_norm(channels):
    c_per_g = 32                      
    num_groups = channels // c_per_g
    # Ensure divisibility
    while channels % num_groups:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, residual=False):
        super().__init__()
        self.residual = residual and (in_channels == out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),               # ← Random dropout
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.residual else out


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, base_dropout=0.2):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 256, dropout=base_dropout)
        self.conv2 = ConvBlock(256,  128, dropout=base_dropout*0.5)
        self.classifier = nn.Conv2d(128, num_classes, 1)

        # Split the ×4 upsampling to reduce artifacts from one-time interpolation
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up4_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, features, original_size):
        x = features[0] if isinstance(features, (list, tuple)) else features

        x = self.conv1(x)
        x = self.up4_1(x)
        x = self.conv2(x)
        x = self.up4(x)

        logits = self.classifier(x)

        return logits


class MultiLevelDecoder2(nn.Module):
    """
    Innovation Point B: Fuse multi-level features
    """

    def __init__(self, in_channels_list, fusion_channels, num_classes):
        super().__init__()
        self.projections = nn.ModuleList()
        for in_channels in in_channels_list:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fusion_channels, kernel_size=1, bias=False),
                    make_group_norm(fusion_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.decoder = SimpleDecoder(fusion_channels, fusion_channels // 2, num_classes)

    def fuse(self, features):
        projected_features = [proj(feat) for proj, feat in zip(self.projections, features)]

        fused_features = projected_features[0]
        for pf in projected_features[1:]:
            fused_features = fused_features + pf


        return fused_features

    def forward(self, features, original_size):
        fused_features = self.fuse(features)
        return self.decoder(fused_features, original_size)


def get_decoder(config, in_channels_list, num_classes):
    decoder_type = config['model']['decoder']['type']
    params = config['model']['decoder'].get('params', {})

    if decoder_type == 'MultiLevel':
        return MultiLevelDecoder2(in_channels_list, params['fusion_channels'], num_classes)
    else:
        raise Exception("not implemented")
