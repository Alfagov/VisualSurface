from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass(slots=True)
class RasterSpec:
    Nu: int
    Nv: int
    u_min: float
    u_max: float
    v_min: float
    v_max: float


@dataclass(slots=True)
class SurfaceBatch:
    img: torch.Tensor
    quote_u: torch.Tensor
    quote_v: torch.Tensor
    quote_num: torch.Tensor
    cp: torch.Tensor
    style: torch.Tensor
    quote_valid: torch.Tensor
    global_feats: torch.Tensor
    quote_iv: torch.Tensor
    K: torch.Tensor
    T_days: torch.Tensor
    r_q: torch.Tensor
    q_q: torch.Tensor
    spot: torch.Tensor

    def to(self, device: torch.device | str) -> "SurfaceBatch":
        moved = {}
        for f in fields(self):
            val = getattr(self, f.name)
            moved[f.name] = val.to(device) if torch.is_tensor(val) else val
        return SurfaceBatch(**moved)

    def pin_memory(self) -> "SurfaceBatch":
        pinned = {}
        for f in fields(self):
            val = getattr(self, f.name)
            pinned[f.name] = val.pin_memory() if torch.is_tensor(val) else val
        return SurfaceBatch(**pinned)

    def validate(self, spec: RasterSpec) -> None:
        b = self.img.shape[0]
        if self.img.ndim != 4:
            raise ValueError(f"img must be [B,5,Nv,Nu], got shape={tuple(self.img.shape)}")
        if self.img.shape[1] != 5 or self.img.shape[2] != spec.Nv or self.img.shape[3] != spec.Nu:
            raise ValueError(
                f"img must be [B,5,{spec.Nv},{spec.Nu}], got shape={tuple(self.img.shape)}"
            )

        if self.quote_u.ndim != 2:
            raise ValueError(f"quote_u must be [B,N], got shape={tuple(self.quote_u.shape)}")

        n = self.quote_u.shape[1]
        expected_bn = {
            "quote_v": self.quote_v,
            "cp": self.cp,
            "style": self.style,
            "quote_valid": self.quote_valid,
            "quote_iv": self.quote_iv,
            "K": self.K,
            "T_days": self.T_days,
            "r_q": self.r_q,
            "q_q": self.q_q,
        }
        for name, tensor in expected_bn.items():
            if tensor.ndim != 2 or tensor.shape[0] != b or tensor.shape[1] != n:
                raise ValueError(f"{name} must be [B,N], got shape={tuple(tensor.shape)}")

        if self.quote_num.ndim != 3 or self.quote_num.shape[0] != b or self.quote_num.shape[1] != n or self.quote_num.shape[2] != 10:
            raise ValueError(f"quote_num must be [B,N,10], got shape={tuple(self.quote_num.shape)}")

        if self.global_feats.ndim != 2 or self.global_feats.shape[0] != b or self.global_feats.shape[1] != 4:
            raise ValueError(f"global_feats must be [B,4], got shape={tuple(self.global_feats.shape)}")

        if self.spot.ndim != 1 or self.spot.shape[0] != b:
            raise ValueError(f"spot must be [B], got shape={tuple(self.spot.shape)}")
