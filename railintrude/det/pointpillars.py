"""Pillar 化 + PointPillars 头（纯 PyTorch，8 GB 卡可跑）。

与原 OpenPCDet 实现的刻意差异（都会写进 docs/readings/D3.md 的偏差表）：
- 点特征 4 维（原文 9 维），pillar 编码用单层 Linear+BN+ReLU+max；
- backbone 卷积保持空间尺寸（p=1），输出为 pillar 网格的 1/2（每个输出格 1.28 m，够放人）；
- 每格 anchor 数 A = 类别数 × 2 朝向 = 8，cls/box/dir 三支输出；
- 输入 batch 约定：pillars (P,M,4)、coords (P,3)=[b, ix, iy]、num_points (P,)。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PillarFeatureNet(nn.Module):
    def __init__(self, cin=4, cout=64):
        super().__init__()
        self.cout = cout
        self.net = nn.Sequential(nn.Linear(cin, cout, bias=False),
                                 nn.BatchNorm1d(cout), nn.ReLU(inplace=True))

    def forward(self, pillars, npts):
        p, m, c = pillars.shape
        if p == 0:
            return pillars.new_zeros(0, self.cout)
        x = self.net(pillars.reshape(p * m, c)).reshape(p, m, self.cout)
        return x.max(dim=1).values                     # (P, C)


class Backbone(nn.Module):
    def __init__(self, cin=64, ch=(64, 128, 256), out_ch=192):
        super().__init__()
        c1, c2, c3 = ch
        self.block1 = nn.Sequential(_conv_bn(cin, c1, s=2), _conv_bn(c1, c1))
        self.block2 = nn.Sequential(_conv_bn(c1, c2, s=1), _conv_bn(c2, c2), _conv_bn(c2, c2))
        self.block3 = nn.Sequential(_conv_bn(c2, c3, s=1), _conv_bn(c3, c3), _conv_bn(c3, c3))
        self.up3 = nn.Sequential(nn.Conv2d(c3, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(True))
        self.out = _conv_bn(c2 * 2, out_ch)

    def forward(self, x):
        x1 = self.block1(x)          # 1/2
        x2 = self.block2(x1)         # 1/4
        x3 = self.block3(x2)         # 1/2（保持尺寸）
        return self.out(torch.cat([self.up3(x3), x2], dim=1))    # 每格 1.28 m


class PointPillars(nn.Module):
    def __init__(self, num_classes=4, garrid_hw=(400, 2400), bev_stride=4,
                 anchors_per_cell=8, feat_ch=64, bev_out=192, pillar_dim=4):
        super().__init__()
        self.num_classes = num_classes
        self.gw, self.gh = garrid_hw[1], garrid_hw[0]    # gw=沿 x 的格数，gh=沿 y 的格数
        self.bev_stride = bev_stride
        self.A = anchors_per_cell                         # 4 类 × 2 朝向
        self.pillar_dim = pillar_dim
        self.pfn = PillarFeatureNet(pillar_dim, feat_ch)
        self.backbone = Backbone(feat_ch, out_ch=bev_out)
        self.head = nn.Sequential(_conv_bn(bev_out, 128),
                                  nn.Conv2d(128, self.A * (num_classes + 7 + 2), 3, 1, 1))
        # focal 先验：把分类通道的偏置初始化成 p=0.01，否则 192k anchor 的负样本损失会到万级
        with torch.no_grad():
            b = self.head[-1].bias.view(self.A, num_classes + 9)
            b[:, :num_classes].fill_(-4.595)

    def forward(self, batch):
        pillars, coords, npts = batch["pillars"], batch["coords"], batch["num_points"]
        bsz = int(batch.get("batch_size", int(coords[:, 0].max()) + 1 if len(coords) else 1))
        feats = self.pfn(pillars, npts)                       # (P, C)
        gx = self.gw // self.bev_stride                       # 沿 x 的格数
        gy = self.gh // self.bev_stride                       # 沿 y 的格数
        canvas = feats.new_zeros(bsz, feats.shape[1], gy, gx)
        if len(coords):
            b = coords[:, 0].long()
            ix = torch.clamp(coords[:, 1].long() // self.bev_stride, 0, gx - 1)
            iy = torch.clamp(coords[:, 2].long() // self.bev_stride, 0, gy - 1)
            canvas[b, :, iy, ix] = feats         # H=沿 y，W=沿 x
        raw = self.head(self.backbone(canvas))
        bsz, _, hb, wb = raw.shape
        # 展平顺序必须与 make_anchors 一致：x 外层、y 中层、anchor 内层
        raw = raw.view(bsz, self.A, self.num_classes + 9, hb, wb)
        raw = raw.permute(0, 4, 3, 1, 2).contiguous()     # (B, gx, gy, A, C)
        raw = raw.view(bsz, wb * hb * self.A, self.num_classes + 9)
        n = wb * hb * self.A
        cls = raw[..., :self.num_classes]
        box = raw[..., self.num_classes:self.num_classes + 7]
        dire = raw[..., self.num_classes + 7:]
        return {"cls_preds": cls, "box_preds": box, "dir_preds": dire, "grid_hw": (hb, wb)}
