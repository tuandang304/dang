"""External state-of-the-art polyp segmentation baselines, ported from the official code.

Each model keeps the official decoder, the official training loss (structure loss on the official
set of side outputs) and the official inference output. Only the backbone loader changes: the
ImageNet weights come from timm instead of the authors' local .pth files.

  PraNet    (Fan et al., MICCAI 2020)  github.com/DengPingFan/PraNet       lib/PraNet_Res2Net.py
            Res2Net-50 v1b 26w4s -> timm res2net50d.in1k (deep stem + avg-down = v1b)
            loss: sum of structure losses on lateral maps 5,4,3,2; inference: lateral_map_2
  Polyp-PVT (Dong et al., CAAI AIR 2023) github.com/DengPingFan/Polyp-PVT lib/pvt.py
            PVTv2-B2 -> timm pvt_v2_b2.in1k
            loss: structure(P1) + structure(P2); inference: P1 + P2
  EMCAD     (Rahman et al., CVPR 2024) github.com/SLDGroup/EMCAD          lib/networks.py
            PVTv2-B2 -> timm pvt_v2_b2.in1k, decoder vendored verbatim (third_party/emcad_decoders.py)
            loss: structure on p4,p3,p2,p1 and on their sum; inference: p1 (= P[-1])
  PraNet-V2 (Hu et al., 2025)        github.com/ai4colonoscopy/PraNet-V2 binary_seg/lib/pranet.py
            PVT_PraNet_V2(num_class=1, use_softmax=True), PVTv2-B2 -> timm pvt_v2_b2.in1k
            loss: sum over lateral maps 2-5 of structure loss + 0.8 * weighted BCE on the background
            maps (background mask = 1 - gt); inference: sum of the four foreground lateral maps

The backbone is stored as `self.cnn` so the shared optimizer gives it the backbone learning rate.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dang.third_party.emcad_decoders import EMCAD


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Official weighted BCE + weighted IoU loss shared by PraNet, Polyp-PVT and EMCAD."""
    pred = pred.float()
    mask = mask.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class BasicConv2d(nn.Module):
    # Official definition: conv + BN, the ReLU is constructed but not applied.
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


# ------------------------------------------------------------------------------------------------
# PraNet
# ------------------------------------------------------------------------------------------------
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x_cat = self.conv_cat(torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1))
        return self.relu(x_cat + self.conv_res(x))


class PraNetAggregation(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        return self.conv5(self.conv4(x3_2))


class PraNetModel(nn.Module):
    def __init__(self, cfg, channel: int = 32):
        super().__init__()
        self.cfg = cfg
        self.cnn = timm.create_model(getattr(cfg, "sota_backbone", "res2net50d.in1k"), pretrained=True,
                                     features_only=True, out_indices=(1, 2, 3, 4))
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.agg1 = PraNetAggregation(channel)
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, gt_mask=None):
        _, x2, x3, x4 = self.cnn(x)
        ra5_feat = self.agg1(self.rfb4_1(x4), self.rfb3_1(x3), self.rfb2_1(x2))
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode="bilinear")

        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode="bilinear")
        x = (-1 * torch.sigmoid(crop_4) + 1).expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        x = self.ra4_conv5(x) + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode="bilinear")

        crop_3 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = (-1 * torch.sigmoid(crop_3) + 1).expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        x = self.ra3_conv4(x) + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode="bilinear")

        crop_2 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = (-1 * torch.sigmoid(crop_2) + 1).expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        x = self.ra2_conv4(x) + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode="bilinear")

        return {"logits": lateral_map_2, "side_outputs": [lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2]}

    @staticmethod
    def official_loss(outputs, masks):
        return sum(structure_loss(p, masks) for p in outputs["side_outputs"])


# ------------------------------------------------------------------------------------------------
# Polyp-PVT
# ------------------------------------------------------------------------------------------------
class CFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        return self.conv4(x3_2)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        # Official code calls the deprecated F.upsample, whose default mode is "nearest".
        edge = F.interpolate(edge, (x.size()[-2], x.size()[-1]), mode="nearest")
        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1.0 / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class PolypPVTModel(nn.Module):
    def __init__(self, cfg, channel: int = 32):
        super().__init__()
        self.cfg = cfg
        self.cnn = timm.create_model(getattr(cfg, "sota_backbone", "pvt_v2_b2.in1k"), pretrained=True,
                                     features_only=True)  # [64, 128, 320, 512]
        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        self.down05 = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x, gt_mask=None):
        x1, x2, x3, x4 = self.cnn(x)
        x1 = self.ca(x1) * x1
        cim_feature = self.sa(x1) * x1
        cfm_feature = self.CFM(self.Translayer4_1(x4), self.Translayer3_1(x3), self.Translayer2_1(x2))
        T2 = self.down05(self.Translayer2_0(cim_feature))
        sam_feature = self.SAM(cfm_feature, T2)
        p1 = F.interpolate(self.out_CFM(cfm_feature), scale_factor=8, mode="bilinear")
        p2 = F.interpolate(self.out_SAM(sam_feature), scale_factor=8, mode="bilinear")
        return {"logits": p1 + p2, "side_outputs": [p1, p2]}

    @staticmethod
    def official_loss(outputs, masks):
        p1, p2 = outputs["side_outputs"]
        return structure_loss(p1, masks) + structure_loss(p2, masks)


# ------------------------------------------------------------------------------------------------
# EMCAD
# ------------------------------------------------------------------------------------------------
class EMCADModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cnn = timm.create_model(getattr(cfg, "sota_backbone", "pvt_v2_b2.in1k"), pretrained=True,
                                     features_only=True)
        channels = [512, 320, 128, 64]
        # Defaults of the official train_polyp.py: kernel_sizes [1,3,5], expansion 2, parallel DW,
        # additive aggregation, LGAG kernel 3, ReLU6 in MSCB.
        self.decoder = EMCAD(channels=channels, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                             add=True, lgag_ks=3, activation="relu6")
        self.out_head4 = nn.Conv2d(channels[0], 1, 1)
        self.out_head3 = nn.Conv2d(channels[1], 1, 1)
        self.out_head2 = nn.Conv2d(channels[2], 1, 1)
        self.out_head1 = nn.Conv2d(channels[3], 1, 1)

    def forward(self, x, gt_mask=None):
        x1, x2, x3, x4 = self.cnn(x)
        dec_outs = self.decoder(x4, [x3, x2, x1])
        p4 = F.interpolate(self.out_head4(dec_outs[0]), scale_factor=32, mode="bilinear")
        p3 = F.interpolate(self.out_head3(dec_outs[1]), scale_factor=16, mode="bilinear")
        p2 = F.interpolate(self.out_head2(dec_outs[2]), scale_factor=8, mode="bilinear")
        p1 = F.interpolate(self.out_head1(dec_outs[3]), scale_factor=4, mode="bilinear")
        return {"logits": p1, "side_outputs": [p4, p3, p2, p1]}

    @staticmethod
    def official_loss(outputs, masks):
        p = outputs["side_outputs"]
        return sum(structure_loss(q, masks) for q in p) + structure_loss(p[0] + p[1] + p[2] + p[3], masks)


# ------------------------------------------------------------------------------------------------
# PraNet-V2 (PVTv2-B2 variant, the stronger of the two official backbones)
# ------------------------------------------------------------------------------------------------
def structure_loss_v2(pred, pred_bg, mask):
    """Official PraNet-V2 loss: structure loss on the foreground plus 0.8 x weighted BCE on the background."""
    pred = pred.float()
    pred_bg = pred_bg.float()
    mask = mask.float()
    mask_bg = 1 - mask
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction="none")
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou + 0.8 * wbce2).mean()


class PraNetV2Aggregation(PraNetAggregation):
    # Official partial decoder with separate foreground and background heads (conv5_fg / conv5_bg).
    def __init__(self, channel, num_class=1):
        super().__init__(channel)
        del self.conv5
        self.conv5_fg = nn.Conv2d(3 * channel, num_class, 1)
        self.conv5_bg = nn.Conv2d(3 * channel, num_class, 1)

    def forward(self, x1, x2, x3):
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        x = self.conv4(x3_2)
        return self.conv5_fg(x), self.conv5_bg(x)


class PraNetV2Model(nn.Module):
    def __init__(self, cfg, channel: int = 32, num_class: int = 1):
        super().__init__()
        self.cfg = cfg
        self.cnn = timm.create_model(getattr(cfg, "sota_backbone", "pvt_v2_b2.in1k"), pretrained=True,
                                     features_only=True)  # [64, 128, 320, 512]
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        self.agg1 = PraNetV2Aggregation(channel, num_class)
        # DSRA3
        self.ra4_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5_fg = BasicConv2d(256, num_class, kernel_size=1)
        self.ra4_conv5_bg = BasicConv2d(256, num_class, kernel_size=1)
        # DSRA2
        self.ra3_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4_fg = BasicConv2d(64, num_class, kernel_size=3, padding=1)
        self.ra3_conv4_bg = BasicConv2d(64, num_class, kernel_size=3, padding=1)
        # DSRA1
        self.ra2_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4_fg = BasicConv2d(64, num_class, kernel_size=3, padding=1)
        self.ra2_conv4_bg = BasicConv2d(64, num_class, kernel_size=3, padding=1)

    @staticmethod
    def _dsra(feat_fg, crop_fg, crop_bg):
        # Official use_softmax=True path; with num_class=1 the channel softmax is identically 1.
        return feat_fg + feat_fg.mul(F.softmax(crop_fg - crop_bg, dim=1))

    def forward(self, x, gt_mask=None):
        _, x2, x3, x4 = self.cnn(x)
        ra5_fg, ra5_bg = self.agg1(self.rfb4_1(x4), self.rfb3_1(x3), self.rfb2_1(x2))
        lm5_fg = F.interpolate(ra5_fg, scale_factor=8, mode="bilinear")
        lm5_bg = F.interpolate(ra5_bg, scale_factor=8, mode="bilinear")

        crop_fg = F.interpolate(ra5_fg, scale_factor=0.25, mode="bilinear")
        crop_bg = F.interpolate(ra5_bg, scale_factor=0.25, mode="bilinear")
        h = self.ra4_conv1(x4)
        h = F.relu(self.ra4_conv2(h))
        h = F.relu(self.ra4_conv3(h))
        h = F.relu(self.ra4_conv4(h))
        ra4_fg = self._dsra(self.ra4_conv5_fg(h), crop_fg, crop_bg)
        ra4_bg = self.ra4_conv5_bg(h)
        lm4_fg = F.interpolate(ra4_fg, scale_factor=32, mode="bilinear")
        lm4_bg = F.interpolate(ra4_bg, scale_factor=32, mode="bilinear")

        crop_fg = F.interpolate(ra4_fg, scale_factor=2, mode="bilinear")
        crop_bg = F.interpolate(ra4_bg, scale_factor=2, mode="bilinear")
        h = self.ra3_conv1(x3)
        h = F.relu(self.ra3_conv2(h))
        h = F.relu(self.ra3_conv3(h))
        ra3_fg = self._dsra(self.ra3_conv4_fg(h), crop_fg, crop_bg)
        ra3_bg = self.ra3_conv4_bg(h)
        lm3_fg = F.interpolate(ra3_fg, scale_factor=16, mode="bilinear")
        lm3_bg = F.interpolate(ra3_bg, scale_factor=16, mode="bilinear")

        crop_fg = F.interpolate(ra3_fg, scale_factor=2, mode="bilinear")
        crop_bg = F.interpolate(ra3_bg, scale_factor=2, mode="bilinear")
        h = self.ra2_conv1(x2)
        h = F.relu(self.ra2_conv2(h))
        h = F.relu(self.ra2_conv3(h))
        ra2_fg = self._dsra(self.ra2_conv4_fg(h), crop_fg, crop_bg)
        ra2_bg = self.ra2_conv4_bg(h)
        lm2_fg = F.interpolate(ra2_fg, scale_factor=8, mode="bilinear")
        lm2_bg = F.interpolate(ra2_bg, scale_factor=8, mode="bilinear")

        fg = [lm2_fg, lm3_fg, lm4_fg, lm5_fg]
        bg = [lm2_bg, lm3_bg, lm4_bg, lm5_bg]
        return {"logits": lm2_fg + lm3_fg + lm4_fg + lm5_fg, "side_outputs": fg, "bg_outputs": bg}

    @staticmethod
    def official_loss(outputs, masks):
        return sum(structure_loss_v2(f, b, masks) for f, b in zip(outputs["side_outputs"], outputs["bg_outputs"]))


SOTA_MODELS = {"pranet": PraNetModel, "polyp_pvt": PolypPVTModel, "emcad": EMCADModel, "pranet_v2": PraNetV2Model}
