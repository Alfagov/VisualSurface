from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from visualsurface.math_ops import make_uv_grid
from visualsurface.types import RasterSpec


class PatchViEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        d_model: int,
        patch: int,
        n_layers: int,
        n_heads: int,
        grid_hw: Tuple[int, int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch, bias=True)

        Nv, Nu = grid_hw
        assert Nv % patch == 0 and Nu % patch == 0, "Nv/Nu must be divisible by patch"
        self.n_patches = (Nv // patch) * (Nu // patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(img)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos
        x = self.enc(x)
        return self.norm(x)


class SurfaceReconstructor(nn.Module):
    def __init__(
        self,
        spec: RasterSpec,
        d_model: int = 256,
        img_in_ch: int = 6,
        patch: int = 4,
        vit_layers: int = 4,
        vit_heads: int = 8,
        quote_num_dim: int = 10,
        global_dim: int = 4,
        dec_layers: int = 4,
        dec_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spec = spec
        self.img_enc = PatchViEncoder(img_in_ch, d_model, patch, vit_layers, vit_heads, (spec.Nv, spec.Nu), dropout)

        self.cp_emb = nn.Embedding(2, d_model)
        self.style_emb = nn.Embedding(2, d_model)

        self.quote_num_mlp = nn.Sequential(
            nn.Linear(2 + quote_num_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=dec_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        img: torch.Tensor,
        quote_u: torch.Tensor,
        quote_v: torch.Tensor,
        quote_num: torch.Tensor,
        cp_flag: torch.Tensor,
        ex_style: torch.Tensor,
        quote_valid: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        B, _ = quote_u.shape
        img_tokens = self.img_enc(img)

        q_in = torch.cat([quote_u.unsqueeze(-1), quote_v.unsqueeze(-1), quote_num], dim=-1)
        qtok = self.quote_num_mlp(q_in) + self.cp_emb(cp_flag) + self.style_emb(ex_style)

        gtok = self.global_mlp(global_feats).unsqueeze(1)
        mem = torch.cat([gtok, qtok, img_tokens], dim=1)
        P = img_tokens.shape[1]
        mem_kpm = torch.cat(
            [
                torch.zeros(B, 1, device=img.device, dtype=torch.bool),
                ~quote_valid,
                torch.zeros(B, P, device=img.device, dtype=torch.bool),
            ],
            dim=1,
        )

        u_vec, v_vec = make_uv_grid(self.spec, device=img.device)
        uu = u_vec.unsqueeze(0).repeat(self.spec.Nv, 1)
        vv = v_vec.unsqueeze(1).repeat(1, self.spec.Nu)
        uv = torch.stack([uu, vv], dim=-1).view(1, -1, 2).repeat(B, 1, 1)

        tgt = self.query_mlp(uv)
        h = self.decoder(tgt=tgt, memory=mem, memory_key_padding_mask=mem_kpm)
        iv = self.head(h).squeeze(-1)
        iv = F.softplus(iv) + 1e-6
        return iv.view(B, self.spec.Nv, self.spec.Nu)
