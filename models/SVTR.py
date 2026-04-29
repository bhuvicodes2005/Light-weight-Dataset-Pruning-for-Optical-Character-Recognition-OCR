"""
SVTR-Tiny: Scene Text Recognition with a Single Visual Model
Paper: https://arxiv.org/abs/2205.00159  (IJCAI 2022)

Drop-in replacement for CRNN / ViTSTR in pretrain.py.
Output: (T, N, num_classes)  — sequence-first, ready for CTCLoss.

Constructor signature matches the other CTC models:
    SVTR(img_channel=1, img_height=32, img_width=128, num_class=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── tiny hyper-params (from Table 1, SVTR paper) ──────────────────────────────
# embed dims per stage, depths, num-heads, mixer types
_SVTR_TINY = dict(
    embed_dims   = [64, 128, 256],
    depths       = [3, 6, 3],
    num_heads    = [2, 4, 8],
    # 'L' = Local (convolutional) mixer, 'G' = Global (self-attention) mixer
    mixer_types  = (
        ['L'] * 3 +          # stage-0
        ['L'] * 3 + ['G'] * 3 +  # stage-1
        ['G'] * 3            # stage-2
    ),
    mlp_ratio    = 4.0,
    drop_rate    = 0.0,
    attn_drop    = 0.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNAct(nn.Sequential):
    """Conv → BN → GELU (or identity if act=False)."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, act=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class PatchEmbed(nn.Module):
    """
    Two-stage stem that maps (N, C, H, W) → (N, L, D).

    Stage-1: 3×3 conv, stride 2  → H/2, W/2
    Stage-2: 3×3 conv, stride 2  → H/4, W/4  (spatial size fed into mixing)

    For the default 32×128 input this gives a 8×32 = 256-token sequence.
    """
    def __init__(self, img_channel, img_height, img_width, embed_dim):
        super().__init__()
        mid = embed_dim // 2
        self.proj = nn.Sequential(
            ConvBNAct(img_channel, mid,        3, stride=2, padding=1),
            ConvBNAct(mid,         embed_dim,  3, stride=2, padding=1),
        )
        self.h_out = img_height  // 4
        self.w_out = img_width   // 4
        self.num_patches = self.h_out * self.w_out

    def forward(self, x):
        # x: (N, C, H, W) → (N, D, H/4, W/4) → (N, H/4*W/4, D)
        x = self.proj(x)
        N, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (N, L, D)
        return x, H, W


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LocalMixer(nn.Module):
    """
    Local (convolutional) mixer: depth-wise 3×3 conv on the 2-D feature map,
    keeping spatial context within a neighbourhood.
    """
    def __init__(self, dim, H, W, kernel=3):
        super().__init__()
        self.H, self.W = H, W
        pad = kernel // 2
        self.dw = nn.Conv2d(dim, dim, kernel, padding=pad, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: (N, L, D),  L = H*W
        N, L, D = x.shape
        x2 = x.transpose(1, 2).reshape(N, D, self.H, self.W)
        x2 = self.bn(self.dw(x2))
        return x2.flatten(2).transpose(1, 2)   # (N, L, D)


class GlobalMixer(nn.Module):
    """Multi-head self-attention (global)."""
    def __init__(self, dim, num_heads, attn_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out


class MixingBlock(nn.Module):
    """One SVTR mixing block: norm → mixer → norm → MLP (pre-norm style)."""
    def __init__(self, dim, num_heads, H, W,
                 mixer_type='L', mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if mixer_type == 'L':
            self.mixer = LocalMixer(dim, H, W)
        else:
            self.mixer = GlobalMixer(dim, num_heads, attn_drop)
        self.mlp = Mlp(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MergingLayer(nn.Module):
    """
    Halve the height via a strided 3×3 conv (height merging as in SVTR).
    Width is preserved so the sequence length stays manageable.
    """
    def __init__(self, in_dim, out_dim, H, W):
        super().__init__()
        # stride (2,1): halve height, keep width
        self.conv = nn.Conv2d(in_dim, out_dim, 3,
                              stride=(2, 1), padding=(1, 1), bias=False)
        self.bn   = nn.BatchNorm2d(out_dim)
        self.act  = nn.GELU()
        self.h_out = max(1, H // 2)
        self.w_out = W

    def forward(self, x, H, W):
        N, L, D = x.shape
        x = x.transpose(1, 2).reshape(N, D, H, W)
        x = self.act(self.bn(self.conv(x)))
        H2, W2 = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, H2, W2


# ─────────────────────────────────────────────────────────────────────────────
# Full SVTR model
# ─────────────────────────────────────────────────────────────────────────────

class SVTR(nn.Module):
    """
    SVTR-Tiny scene text recogniser.

    Output shape: (T, N, num_classes)  — sequence-first for nn.CTCLoss.
    T = W/4  (width after the patch-embed stem, height fully collapsed).
    """

    def __init__(
        self,
        img_channel: int = 1,
        img_height:  int = 32,
        img_width:   int = 128,
        num_class:   int = 64,
        cfg: dict = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = _SVTR_TINY

        embed_dims  = cfg['embed_dims']   # e.g. [64, 128, 256]
        depths      = cfg['depths']       # e.g. [3, 6, 3]
        num_heads   = cfg['num_heads']    # e.g. [2, 4, 8]
        mixer_types = cfg['mixer_types']  # flat list, len = sum(depths)
        mlp_ratio   = cfg['mlp_ratio']
        drop        = cfg['drop_rate']
        attn_drop   = cfg['attn_drop']

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_channel, img_height, img_width,
                                      embed_dims[0])
        H0 = self.patch_embed.h_out   # img_height // 4
        W0 = self.patch_embed.w_out   # img_width  // 4

        # ── Stage 0 ───────────────────────────────────────────────────────────
        mix_idx = 0
        self.blocks0 = nn.ModuleList([
            MixingBlock(embed_dims[0], num_heads[0], H0, W0,
                        mixer_types[mix_idx + i], mlp_ratio, drop, attn_drop)
            for i in range(depths[0])
        ])
        mix_idx += depths[0]

        # merge: H0 → H1 = H0//2,  D0 → D1
        self.merge0 = MergingLayer(embed_dims[0], embed_dims[1], H0, W0)
        H1 = self.merge0.h_out

        # ── Stage 1 ───────────────────────────────────────────────────────────
        self.blocks1 = nn.ModuleList([
            MixingBlock(embed_dims[1], num_heads[1], H1, W0,
                        mixer_types[mix_idx + i], mlp_ratio, drop, attn_drop)
            for i in range(depths[1])
        ])
        mix_idx += depths[1]

        # merge: H1 → H2 = H1//2,  D1 → D2
        self.merge1 = MergingLayer(embed_dims[1], embed_dims[2], H1, W0)
        H2 = self.merge1.h_out

        # ── Stage 2 ───────────────────────────────────────────────────────────
        self.blocks2 = nn.ModuleList([
            MixingBlock(embed_dims[2], num_heads[2], H2, W0,
                        mixer_types[mix_idx + i], mlp_ratio, drop, attn_drop)
            for i in range(depths[2])
        ])

        # ── Combining layer: collapse height → 1 via average, project D2 → D1 ─
        # (SVTR paper §3.3 "Combining" — avg-pool height, linear projection)
        self.combine_norm = nn.LayerNorm(embed_dims[2])
        self.combine_proj = nn.Linear(embed_dims[2] * H2, embed_dims[1])
        self.combine_act  = nn.GELU()
        self._H2 = H2
        self._W0 = W0

        # ── CTC head ─────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dims[1])
        self.head = nn.Linear(embed_dims[1], num_class)

        self._init_weights()

    # ── weight init ──────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W)  — grayscale: C=1, H=32, W=128
        Returns:
            (T, N, num_classes)  — T = W/4 = 32 for default input
        """
        # ── Patch embed: (N,C,H,W) → (N,L,D0), L=H0*W0 ──────────────────────
        x, H, W = self.patch_embed(x)

        # ── Stage 0 ───────────────────────────────────────────────────────────
        for blk in self.blocks0:
            x = blk(x)
        x, H, W = self.merge0(x, H, W)   # H halved

        # ── Stage 1 ───────────────────────────────────────────────────────────
        for blk in self.blocks1:
            x = blk(x)
        x, H, W = self.merge1(x, H, W)   # H halved again

        # ── Stage 2 ───────────────────────────────────────────────────────────
        for blk in self.blocks2:
            x = blk(x)

        # ── Combining: collapse height dimension ──────────────────────────────
        # x: (N, H2*W, D2)
        N, L, D2 = x.shape
        H2, W_tok = self._H2, self._W0
        x = self.combine_norm(x)
        # reshape to (N, W, H2*D2) and project
        x = x.reshape(N, H2, W_tok, D2)          # (N, H2, W, D2)
        x = x.permute(0, 2, 1, 3)                 # (N, W, H2, D2)
        x = x.reshape(N, W_tok, H2 * D2)          # (N, W, H2*D2)
        x = self.combine_act(self.combine_proj(x)) # (N, W, D1)

        # ── CTC head: (N, W, D1) → (W, N, num_classes) ───────────────────────
        x = self.norm(x)
        x = self.head(x)              # (N, W, num_classes)
        x = x.permute(1, 0, 2)       # (T, N, num_classes),  T = W = W_orig/4
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = SVTR(img_channel=1, img_height=32, img_width=128, num_class=64)
    total = sum(p.numel() for p in model.parameters())
    print(f'SVTR-Tiny parameters: {total / 1e6:.2f}M')

    dummy = torch.randn(4, 1, 32, 128)
    out   = model(dummy)
    print(f'Input : {tuple(dummy.shape)}')
    print(f'Output: {tuple(out.shape)}   (expect (32, 4, 64))')
    assert out.shape == (32, 4, 64), f'Unexpected output shape: {out.shape}'
    print('Shape check passed ✓')