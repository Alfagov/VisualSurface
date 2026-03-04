from __future__ import annotations

import warnings
from typing import Dict

import lightning as l
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from visualsurface.math_ops import (
    bs_call_from_fwd,
    build_term_structure_by_t_days,
    linear_interpolate_1d,
    make_uv_grid,
    no_arb_penalty_from_call_prices,
    sample_iv_grid_at_quotes,
    smoothness_loss_total_variance,
    v_to_t_years,
)
from visualsurface.model import SurfaceReconstructor
from visualsurface.types import RasterSpec, SurfaceBatch


class LitSurfaceModel(l.LightningModule):
    def __init__(
        self,
        spec: RasterSpec,
        lr: float = 2e-4,
        weight_decay: float = 1e-2,
        w_fit: float = 1.0,
        w_smooth: float = 0.1,
        w_arb: float = 0.1,
        hidden_size: int = 256,
        mlp_size: int | None = None,
        d_model: int | None = None,
        patch: int = 4,
        vit_layers: int = 4,
        vit_heads: int = 8,
        dec_layers: int = 4,
        dec_heads: int = 8,
        dropout: float = 0.0,
        vis_every_n_epochs: int = 1,
    ):
        super().__init__()
        if d_model is not None and hidden_size != 256 and hidden_size != d_model:
            raise ValueError("Provide either hidden_size or d_model, or set both to the same value.")
        if d_model is not None:
            hidden_size = d_model
        if mlp_size is None:
            mlp_size = 4 * hidden_size
        if hidden_size <= 0 or mlp_size <= 0:
            raise ValueError("hidden_size and mlp_size must be positive.")

        self.save_hyperparameters(ignore=["spec"])
        self.spec = spec

        self.model = SurfaceReconstructor(
            spec=spec,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            img_in_ch=5,
            patch=patch,
            vit_layers=vit_layers,
            vit_heads=vit_heads,
            quote_num_dim=9,
            global_dim=4,
            dec_layers=dec_layers,
            dec_heads=dec_heads,
            dropout=dropout,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.trainer.max_epochs))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def _compute_losses(self, iv_grid: torch.Tensor, batch: SurfaceBatch) -> Dict[str, torch.Tensor]:
        spec = self.spec
        device = iv_grid.device

        quote_u = batch.quote_u
        quote_v = batch.quote_v
        quote_iv = batch.quote_iv
        quote_valid = batch.quote_valid

        spot = batch.spot
        T_days = batch.T_days
        r_q = batch.r_q
        q_q = batch.q_q

        pred_at_quotes = sample_iv_grid_at_quotes(iv_grid, quote_u, quote_v, spec)
        fit = ((pred_at_quotes - quote_iv)[quote_valid] ** 2).mean()

        _, v_vec = make_uv_grid(spec, device=device)
        T_vec = torch.clamp(v_to_t_years(v_vec), min=1/365)
        smooth = smoothness_loss_total_variance(iv_grid, T_vec, p=2)

        T_u_days, r_u, q_u, M_valid = build_term_structure_by_t_days(T_days, r_q, q_q, quote_valid)
        r_grid = torch.zeros(iv_grid.shape[0], spec.Nv, device=device)
        q_grid = torch.zeros(iv_grid.shape[0], spec.Nv, device=device)

        for b in range(iv_grid.shape[0]):
            Tu_d = T_u_days[b][M_valid[b]].float()
            ru = r_u[b][M_valid[b]]
            qu = q_u[b][M_valid[b]]
            Tu = torch.clamp(Tu_d / 365.0, min=1e-6)
            if Tu.numel() == 1:
                r_grid[b] = ru[0]
                q_grid[b] = qu[0]
            else:
                r_grid[b] = linear_interpolate_1d(T_vec, Tu, ru)
                q_grid[b] = linear_interpolate_1d(T_vec, Tu, qu)

        u_vec, _ = make_uv_grid(spec, device=device)
        T = T_vec.view(1, spec.Nv, 1)
        S = spot.view(-1, 1, 1)
        rr = r_grid.view(-1, spec.Nv, 1)
        qq = q_grid.view(-1, spec.Nv, 1)
        Fwd = S * torch.exp((rr - qq) * T)
        Disc = torch.exp(-rr * T)
        K_grid = Fwd * torch.exp(u_vec.view(1, 1, spec.Nu))

        call = bs_call_from_fwd(Fwd, K_grid, iv_grid, T, Disc)
        arb = no_arb_penalty_from_call_prices(call)

        total = self.hparams.w_fit * fit + self.hparams.w_smooth * smooth + self.hparams.w_arb * arb
        return {"total": total, "fit": fit, "smooth": smooth, "arb": arb}

    def training_step(self, batch: SurfaceBatch, batch_idx: int):
        batch = batch.to(self.device)

        iv_grid = self.model(
            batch.img,
            batch.quote_u,
            batch.quote_v,
            batch.quote_num,
            batch.cp,
            batch.style,
            batch.quote_valid,
            batch.global_feats,
        )
        losses = self._compute_losses(iv_grid, batch)

        self.log("train/total", losses["total"], prog_bar=True)
        self.log("train/fit", losses["fit"], prog_bar=True)
        self.log("train/smooth", losses["smooth"], prog_bar=True)
        self.log("train/arb", losses["arb"], prog_bar=True)
        return losses["total"]

    def validation_step(self, batch: SurfaceBatch, batch_idx: int):
        batch = batch.to(self.device)

        iv_grid = self.model(
            batch.img,
            batch.quote_u,
            batch.quote_v,
            batch.quote_num,
            batch.cp,
            batch.style,
            batch.quote_valid,
            batch.global_feats,
        )
        losses = self._compute_losses(iv_grid, batch)

        self.log("val/total", losses["total"], prog_bar=True)
        self.log("val/fit", losses["fit"])
        self.log("val/smooth", losses["smooth"])
        self.log("val/arb", losses["arb"])

        should_vis = (
            batch_idx == 0
            and (self.current_epoch + 1) % self.hparams.vis_every_n_epochs == 0
            and self.logger is not None
            and hasattr(self.logger, "experiment")
        )
        if should_vis:
            self._log_visualizations(batch, iv_grid)

    def _log_visualizations(self, batch: SurfaceBatch, iv_grid: torch.Tensor) -> None:
        from visualsurface.viz import (
            extract_encoder_attention,
            plot_encoder_attention,
            plot_iv_surface,
            plot_residuals,
            plot_rasterized_input,
        )

        b = 0
        step = self.global_step
        exp = self.logger.experiment

        def _log_image(tag: str, tensor: torch.Tensor) -> None:
            if hasattr(exp, "add_image"):
                # TensorBoard: expects [3, H, W] float in [0, 1]
                exp.add_image(tag, tensor, step)
            else:
                # WandB: expects [H, W, 3] uint8 numpy array
                import wandb
                arr = tensor.permute(1, 2, 0).mul(255).byte().numpy()
                exp.log({tag: wandb.Image(arr)})

        try:
            tensor = plot_rasterized_input(batch.img[b], self.spec)
            _log_image("vis/rasterized_input", tensor)
        except Exception as e:
            warnings.warn(f"vis/rasterized_input failed: {e}")

        try:
            tensor = plot_iv_surface(
                iv_grid[b].detach(),
                self.spec,
                batch.quote_u[b],
                batch.quote_v[b],
                batch.quote_iv[b],
                batch.quote_valid[b],
            )
            _log_image("vis/iv_surface", tensor)
        except Exception as e:
            warnings.warn(f"vis/iv_surface failed: {e}")

        try:
            tensor = plot_residuals(
                iv_grid[b].detach(),
                batch.quote_u[b],
                batch.quote_v[b],
                batch.quote_iv[b],
                batch.quote_valid[b],
                self.spec,
            )
            _log_image("vis/residuals", tensor)
        except Exception as e:
            warnings.warn(f"vis/residuals failed: {e}")

        try:
            attn = extract_encoder_attention(batch.img[[b]], self.model.img_enc)
            tensor = plot_encoder_attention(attn[0], self.spec, self.hparams.patch)
            _log_image("vis/encoder_attention", tensor)
        except Exception as e:
            warnings.warn(f"vis/encoder_attention failed: {e}")
