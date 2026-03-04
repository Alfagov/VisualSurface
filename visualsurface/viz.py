from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from visualsurface.math_ops import make_uv_grid, sample_iv_grid_at_quotes, v_to_t_years
from visualsurface.types import RasterSpec


def _fig_to_tensor(fig) -> torch.Tensor:
    """Convert matplotlib figure to [3, H, W] float32 tensor via buffer_rgba (no PIL needed)."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # drop alpha
    plt.close(fig)
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0


def plot_rasterized_input(img: torch.Tensor, spec: RasterSpec) -> torch.Tensor:
    """5-panel row: IV, occupancy, liquidity, |delta|, gamma.

    img: [5, Nv, Nu]
    Returns [3, H, W] tensor for TensorBoard.
    """
    u_vec = torch.linspace(spec.u_min, spec.u_max, spec.Nu)
    v_vec = torch.linspace(spec.v_min, spec.v_max, spec.Nv)
    T_years = v_to_t_years(v_vec).numpy()
    u_arr = u_vec.numpy()
    extent = [u_arr[0], u_arr[-1], T_years[0], T_years[-1]]

    titles = ["IV", "Occupancy", "Liquidity", "|Delta|", "Gamma"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, title) in enumerate(zip(axes, titles)):
        data = img[i].cpu().numpy()
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("log(K/Fwd)")
        ax.set_ylabel("T (years)")
    fig.tight_layout()
    return _fig_to_tensor(fig)


def plot_iv_surface(
    iv_grid: torch.Tensor,
    spec: RasterSpec,
    quote_u: torch.Tensor,
    quote_v: torch.Tensor,
    quote_iv: torch.Tensor,
    quote_valid: torch.Tensor,
) -> torch.Tensor:
    """2D heatmap of predicted surface with actual quotes overlaid.

    iv_grid: [Nv, Nu]
    quote_*: [N] (single sample)
    Returns [3, H, W] tensor.
    """
    u_arr = torch.linspace(spec.u_min, spec.u_max, spec.Nu).numpy()
    v_arr = torch.linspace(spec.v_min, spec.v_max, spec.Nv)
    T_years = v_to_t_years(v_arr).numpy()
    extent = [u_arr[0], u_arr[-1], T_years[0], T_years[-1]]

    iv_np = iv_grid.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(iv_np, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    plt.colorbar(im, ax=ax, label="IV")

    valid = quote_valid.cpu()
    qu = quote_u[valid].cpu().numpy()
    qv = quote_v[valid].cpu().numpy()
    qiv = quote_iv[valid].cpu().numpy()
    qT = np.exp(qv)

    if qu.size > 0:
        sc = ax.scatter(
            qu, qT, c=qiv, cmap="plasma", s=10,
            edgecolors="white", linewidths=0.3,
            vmin=iv_np.min(), vmax=iv_np.max(),
        )
        plt.colorbar(sc, ax=ax, label="Quote IV")

    ax.axvline(0, color="white", linestyle="--", linewidth=1.0)
    ax.set_title("Predicted IV Surface + Quotes")
    ax.set_xlabel("log(K/Fwd)")
    ax.set_ylabel("T (years)")
    fig.tight_layout()
    return _fig_to_tensor(fig)


def plot_residuals(
    iv_grid: torch.Tensor,
    quote_u: torch.Tensor,
    quote_v: torch.Tensor,
    quote_iv: torch.Tensor,
    quote_valid: torch.Tensor,
    spec: RasterSpec,
) -> torch.Tensor:
    """Scatter: pred−actual IV at quote locations, colored by error (RdBu).

    iv_grid: [Nv, Nu]
    quote_*: [N]
    Returns [3, H, W] tensor.
    """
    pred = sample_iv_grid_at_quotes(
        iv_grid.unsqueeze(0),
        quote_u.unsqueeze(0),
        quote_v.unsqueeze(0),
        spec,
    )[0].detach().cpu()

    valid = quote_valid.cpu()
    res_all = (pred - quote_iv.cpu())
    res_valid = res_all[valid].numpy()
    qu = quote_u[valid].cpu().numpy()
    qv = quote_v[valid].cpu().numpy()
    qT = np.exp(qv)

    rmse = float(np.sqrt((res_valid ** 2).mean())) if res_valid.size > 0 else 0.0
    vmax = float(np.abs(res_valid).max()) if res_valid.size > 0 else 1e-4
    vmax = max(vmax, 1e-4)

    fig, ax = plt.subplots(figsize=(8, 6))
    if qu.size > 0:
        sc = ax.scatter(qu, qT, c=res_valid, cmap="RdBu_r", s=15, vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="Pred − Actual IV")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"Residuals  RMSE={rmse:.4f}")
    ax.set_xlabel("log(K/Fwd)")
    ax.set_ylabel("T (years)")
    fig.tight_layout()
    return _fig_to_tensor(fig)


def extract_encoder_attention(img: torch.Tensor, img_enc: nn.Module) -> torch.Tensor:
    """Run img through img_enc, capturing attention from the last encoder layer.

    Monkey-patches last_layer.self_attn.forward to force need_weights=True,
    average_attn_weights=False, then restores the original in a finally block.

    img: [B, 5, Nv, Nu]
    Returns: [B, n_heads, Np, Np]
    """
    last_layer = img_enc.enc.layers[-1]
    orig_forward = last_layer.self_attn.forward
    captured: list[torch.Tensor] = []

    def _patched(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        out, weights = orig_forward(query, key, value, **kwargs)
        captured.append(weights.detach().cpu())
        return out, weights

    try:
        last_layer.self_attn.forward = _patched
        with torch.no_grad():
            img_enc(img)
    finally:
        last_layer.self_attn.forward = orig_forward

    return captured[0]  # [B, n_heads, Np, Np]


def plot_encoder_attention(
    attn: torch.Tensor,
    spec: RasterSpec,
    patch: int,
) -> torch.Tensor:
    """Per-head heatmaps of mean received attention, reshaped to patch grid.

    attn: [n_heads, Np, Np]
    Returns [3, H, W] tensor.
    """
    n_heads = attn.shape[0]
    Nv_p = spec.Nv // patch
    Nu_p = spec.Nu // patch

    # Mean received attention: average over query dim → [n_heads, Np] → [n_heads, Nv_p, Nu_p]
    mean_recv = attn.mean(dim=1).numpy()
    maps = mean_recv.reshape(n_heads, Nv_p, Nu_p)

    ncols = min(4, n_heads)
    nrows = (n_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for h in range(n_heads):
        row, col = h // ncols, h % ncols
        ax = axes[row][col]
        im = ax.imshow(maps[h], origin="lower", cmap="hot", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Head {h}")
        ax.set_xticks([])
        ax.set_yticks([])

    for h in range(n_heads, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")

    fig.suptitle("Last Encoder Layer — Mean Received Attention")
    fig.tight_layout()
    return _fig_to_tensor(fig)
