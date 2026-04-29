import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightBackbone(nn.Module):
    """Lightweight CNN backbone (~1.2M params)"""
    def __init__(self, img_channel=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = self._make_layer(64, 128, 2, stride=2)
        self.conv3 = self._make_layer(128, 256, 2, stride=2)
        self.conv4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride=2):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # (N, 64,  32, 128)
        x = self.conv2(x)   # (N, 128, 16,  64)
        x = self.conv3(x)   # (N, 256,  8,  32)
        x = self.conv4(x)   # (N, 512,  4,  16)
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=8, max_w=32):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2) * 0.02)

    def forward(self, x):
        N, C, H, W = x.shape
        pos_row = self.row_embed[:H].unsqueeze(1).expand(H, W, C // 2)
        pos_col = self.col_embed[:W].unsqueeze(0).expand(H, W, C // 2)
        pos = torch.cat([pos_row, pos_col], dim=-1)          # (H, W, C)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)


class MaskedLanguageModule(nn.Module):
    """
    MLM: Generates character-wise masks using weakly-supervised learning.
    Only used during training (Language-Aware phase).

    """
    def __init__(self, d_model=512, max_len=25):
        super().__init__()
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.query_proj  = nn.Linear(d_model, d_model)
        self.mask_head   = nn.Sequential(
            nn.Conv2d(d_model, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_feat, char_idx):
        N, C, H, W = visual_feat.shape
        pos_emb  = self.pos_embed(char_idx)                    # (N, C)
        query    = self.query_proj(pos_emb).view(N, C, 1, 1)

        att_scores  = (query * visual_feat).sum(dim=1, keepdim=True)
        att_weights = torch.sigmoid(att_scores)

        mask_map    = self.mask_head(visual_feat * att_weights) # (N, 1, H, W)
        masked_feat = visual_feat * (1 - 0.7 * mask_map)

        return mask_map, masked_feat


class VisualReasoningModule(nn.Module):
    """
    VRM: Uses transformer to reason over visual features.

    """
    def __init__(self, d_model=512, nhead=8, num_layers=2,
                 max_seq_len=25, num_classes=64):
        super().__init__()
        self.d_model    = d_model
        self.num_classes = num_classes

        # After backbone: (N, 512, 4, 16) → flatten spatial → (N, 64, 512)
        max_spatial = 4 * 16          # H * W from backbone output at 32×128 input
        self.pos_embed = nn.Parameter(torch.randn(1, max_spatial, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj  = nn.Linear(d_model, num_classes)

    def forward(self, visual_feat, max_length=25):
        N, C, H, W = visual_feat.shape

        x = visual_feat.flatten(2).permute(0, 2, 1)           # (N, H*W, C)

        seq_len = x.size(1)
        if self.pos_embed.size(1) >= seq_len:
            pos = self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate positional embeddings if the spatial size is larger
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=seq_len, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        x = x + pos
        x = self.transformer(x)                                # (N, seq_len, C)
        logits = self.output_proj(x)                           # (N, seq_len, num_classes)

        logits = F.adaptive_avg_pool1d(
            logits.permute(0, 2, 1),   # (N, num_classes, seq_len)
            max_length
        ).permute(0, 2, 1)            # (N, max_length, num_classes)

        return logits


class VisionLAN(nn.Module):
    """
    VisionLAN: Endows a vision model with language capability via masking.

    Two-phase training:
      1. Language-Free  (LF): MLM disabled — pure visual learning.
      2. Language-Aware (LA): MLM enabled — learns linguistic context.

    """
    def __init__(self, img_channel=1, img_height=32, img_width=128,
                 num_classes=64, max_label_length=25):
        super().__init__()

        self.max_label_length = max_label_length
        self.lf_phase = True          # Start in Language-Free phase

        self.backbone = LightweightBackbone(img_channel)
        self.pos_enc  = PositionalEncoding2D(512, max_h=8, max_w=32)
        self.mlm      = MaskedLanguageModule(d_model=512, max_len=max_label_length)
        self.vrm      = VisualReasoningModule(
            d_model=512, nhead=8, num_layers=2,
            max_seq_len=max_label_length, num_classes=num_classes
        )

        self._print_params()

    # ------------------------------------------------------------------
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[VisionLAN] Total parameters: {total / 1e6:.2f}M")

    def set_lf_phase(self, lf_phase=True):
        """True = Language-Free (no masking), False = Language-Aware (masking)."""
        self.lf_phase  = lf_phase
        phase_name     = "Language-Free" if lf_phase else "Language-Aware"
        print(f"[VisionLAN] Phase: {phase_name}")

    # ------------------------------------------------------------------
    def forward(self, x, max_length=None, label_lengths=None):
        """
        Args:
            x             : (N, C, H, W) input images.
            max_length    : decode length; defaults to self.max_label_length.
            label_lengths : (N,) int tensor with the *actual* character count
                            of each label.  Required during LA-phase training
                            to avoid masking padding positions.
                            If None, falls back to uniform sampling over
                            [0, max_label_length).
        """
        if max_length is None:
            max_length = self.max_label_length

        feat = self.backbone(x)
        feat = feat + self.pos_enc(feat)

        # Apply MLM only in Language-Aware phase during training
        if self.training and not self.lf_phase:
            N = feat.size(0)

            # FIX: Sample masked position within the real label length so that
            # we never mask a padding slot (which carries no visual signal).
            if label_lengths is not None:
                valid_lens = label_lengths.to(feat.device).float().clamp(min=1)
            else:
                valid_lens = torch.full((N,), max_length,
                                        dtype=torch.float, device=feat.device)

            char_idx = (torch.rand(N, device=feat.device) * valid_lens).long()
            _, feat  = self.mlm(feat, char_idx)

        logits = self.vrm(feat, max_length)
        return logits