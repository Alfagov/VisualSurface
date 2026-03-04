from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from visualsurface.types import RasterSpec


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_call_from_fwd(
    Fwd: torch.Tensor,
    K: torch.Tensor,
    vol: torch.Tensor,
    T: torch.Tensor,
    disc: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    sqrtT = torch.sqrt(torch.clamp(T, min=eps))
    sig = torch.clamp(vol, min=eps)
    d1 = (torch.log(torch.clamp(Fwd, min=eps) / torch.clamp(K, min=eps)) + 0.5 * sig * sig * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    return disc * (Fwd * norm_cdf(d1) - K * norm_cdf(d2))


def make_uv_grid(spec: RasterSpec, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    u_vec = torch.linspace(spec.u_min, spec.u_max, spec.Nu, device=device)
    v_vec = torch.linspace(spec.v_min, spec.v_max, spec.Nv, device=device)
    return u_vec, v_vec


def no_arb_penalty_from_call_prices(call: torch.Tensor) -> torch.Tensor:
    mono = F.relu(call[:, :, 1:] - call[:, :, :-1])
    mono_loss = (mono ** 2).mean()

    bf = call[:, :, :-2] - 2 * call[:, :, 1:-1] + call[:, :, 2:]
    convex = F.relu(-bf)
    convex_loss = (convex ** 2).mean()

    cal = call[:, 1:, :] - call[:, :-1, :]
    calendar = F.relu(-cal)
    calendar_loss = (calendar ** 2).mean()

    return mono_loss + convex_loss + calendar_loss


def v_to_t_years(v_vec: torch.Tensor) -> torch.Tensor:
    return torch.exp(v_vec)


def uv_to_normalized_grid(u: torch.Tensor, v: torch.Tensor, spec: RasterSpec) -> torch.Tensor:
    x = 2.0 * (u - spec.u_min) / (spec.u_max - spec.u_min) - 1.0
    y = 2.0 * (v - spec.v_min) / (spec.v_max - spec.v_min) - 1.0
    return torch.stack([x, y], dim=-1)


def sample_iv_grid_at_quotes(
    iv_grid: torch.Tensor,
    quote_u: torch.Tensor,
    quote_v: torch.Tensor,
    spec: RasterSpec,
) -> torch.Tensor:
    img = iv_grid.unsqueeze(1)
    coords = uv_to_normalized_grid(quote_u, quote_v, spec)
    grid = coords.unsqueeze(2)
    sample = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
    return sample[:, 0, :, 0]


def linear_interpolate_1d(xq: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xq_c = torch.clamp(xq, x[0], x[-1])
    idx = torch.searchsorted(x, xq_c, right=False)
    idx = torch.clamp(idx, 1, x.numel() - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    w = (xq_c - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + w * (y1 - y0)


def build_term_structure_by_t_days(
    T_days: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, _ = T_days.shape
    device = T_days.device

    all_Tu, all_ru, all_qu = [], [], []
    maxM = 1

    for b in range(B):
        m = valid[b]
        Tb = T_days[b][m].round().long()
        rb = r[b][m]
        qb = q[b][m]

        if Tb.numel() == 0:
            Tu = torch.tensor([30], device=device, dtype=torch.long)
            ru = torch.tensor([0.0], device=device)
            qu = torch.tensor([0.0], device=device)
        else:
            Tu_vals = torch.unique(Tb)
            Tu_vals, _ = torch.sort(Tu_vals)
            ru_vals, qu_vals = [], []
            for td in Tu_vals:
                mm = Tb == td
                ru_vals.append(rb[mm].median())
                qu_vals.append(qb[mm].median())
            Tu = Tu_vals
            ru = torch.stack(ru_vals)
            qu = torch.stack(qu_vals)

        maxM = max(maxM, Tu.numel())
        all_Tu.append(Tu)
        all_ru.append(ru)
        all_qu.append(qu)

    T_u = torch.zeros(B, maxM, device=device, dtype=torch.long)
    r_u = torch.zeros(B, maxM, device=device)
    q_u = torch.zeros(B, maxM, device=device)
    M_valid = torch.zeros(B, maxM, device=device, dtype=torch.bool)

    for b in range(B):
        m = all_Tu[b].numel()
        T_u[b, :m] = all_Tu[b]
        r_u[b, :m] = all_ru[b]
        q_u[b, :m] = all_qu[b]
        M_valid[b, :m] = True

    return T_u, r_u, q_u, M_valid


def smoothness_loss_total_variance(iv_grid: torch.Tensor, T_vec: torch.Tensor, p: int = 2) -> torch.Tensor:
    _, Nv, _ = iv_grid.shape
    T = T_vec.view(1, Nv, 1).to(iv_grid.device)
    w = (iv_grid ** 2) * torch.clamp(T, min=1e-8)

    d2u = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
    d2v = w[:, 2:, :] - 2 * w[:, 1:-1, :] + w[:, :-2, :]

    if p == 1:
        return d2u.abs().mean() + d2v.abs().mean()
    return (d2u ** 2).mean() + (d2v ** 2).mean()


def rasterize_quotes(
    quote_u: torch.Tensor,
    quote_v: torch.Tensor,
    feat: torch.Tensor,
    valid: torch.Tensor,
    spec: RasterSpec,
    feat_ix: Dict[str, int],
    eps: float = 1e-8,
) -> torch.Tensor:
    device = quote_u.device
    B, N = quote_u.shape
    Nu, Nv = spec.Nu, spec.Nv
    HW = Nu * Nv

    iu = ((quote_u - spec.u_min) / (spec.u_max - spec.u_min) * (Nu - 1)).round().long().clamp(0, Nu - 1)
    iv = ((quote_v - spec.v_min) / (spec.v_max - spec.v_min) * (Nv - 1)).round().long().clamp(0, Nv - 1)

    batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    mask = valid
    flat_idx = (batch_ids[mask] * HW + (iv[mask] * Nu + iu[mask])).long()

    C = 5
    out = torch.zeros(B, C, HW, device=device)
    counts = torch.zeros(B, HW, device=device)

    ones = torch.ones_like(flat_idx, dtype=out.dtype, device=device)
    counts.reshape(-1).scatter_add_(0, flat_idx, ones)

    def _scatter(values: torch.Tensor) -> torch.Tensor:
        buf = torch.zeros(B * HW, device=device)
        buf.scatter_add_(0, flat_idx, values)
        return buf.view(B, HW)

    iv_obs = feat[..., feat_ix["Impl_Vol"]]
    out[:, 0, :] = _scatter(iv_obs[mask])

    bid = feat[..., feat_ix["Bid"]]
    ask = feat[..., feat_ix["Ask"]]
    spread = torch.clamp(ask - bid, min=eps)
    liq = 1.0 / spread
    out[:, 2, :] = _scatter(liq[mask])

    delta = feat[..., feat_ix["delta"]].abs()
    out[:, 3, :] = _scatter(delta[mask])

    gamma = feat[..., feat_ix["gamma"]]
    out[:, 4, :] = _scatter(gamma[mask])

    denom = torch.clamp(counts, min=1.0).unsqueeze(1)
    for ch in [0, 2, 3, 4]:
        out[:, ch, :] = out[:, ch, :] / denom[:, 0, :]

    out[:, 1, :] = (counts > 0).float()
    return out.view(B, C, Nv, Nu)
