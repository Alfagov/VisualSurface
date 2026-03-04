from pathlib import Path

import lightning as l
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from visualsurface import LitSurfaceModel, SurfaceDataModule
from visualsurface.math_ops import make_uv_grid, v_to_t_years


def visualize_surface_for_one_day(
    lit: LitSurfaceModel,
    dm: SurfaceDataModule,
    out_path: str = "surface_one_day.png",
    sample_index: int = 0,
) -> Path:
    if dm.spec is None:
        raise RuntimeError("Data module spec is not initialized.")

    dataset = dm.val_ds if dm.val_ds is not None and len(dm.val_ds) > 0 else dm.train_ds
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("No grouped day data available for visualization.")

    sample_index = min(max(sample_index, 0), len(dataset) - 1)
    row = dataset[sample_index]
    day_label = str(row["date"]) if "date" in row else f"sample_{sample_index}"

    batch = dm.collate_fn([row]).to(lit.device)
    lit.eval()
    with torch.no_grad():
        iv_grid = lit.model(
            batch.img,
            batch.quote_u,
            batch.quote_v,
            batch.quote_num,
            batch.cp,
            batch.style,
            batch.quote_valid,
            batch.global_feats,
        )[0]

    u_vec, v_vec = make_uv_grid(dm.spec, device=iv_grid.device)
    k_over_fwd = torch.exp(u_vec)
    t_days = v_to_t_years(v_vec) * 365.0

    valid = batch.quote_valid[0]
    obs_k_over_fwd = torch.exp(batch.quote_u[0][valid]).detach().cpu().numpy()
    obs_t_days = (torch.exp(batch.quote_v[0][valid]) * 365.0).detach().cpu().numpy()
    obs_iv = batch.quote_iv[0][valid].detach().cpu().numpy()

    surf = iv_grid.detach().cpu().numpy()
    x = k_over_fwd.detach().cpu().numpy()
    y = t_days.detach().cpu().numpy()

    yy, xx = torch.meshgrid(t_days.detach().cpu(), k_over_fwd.detach().cpu(), indexing="ij")

    fig = plt.figure(figsize=(14, 6))

    ax0 = fig.add_subplot(1, 2, 1)
    im = ax0.imshow(
        surf,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        cmap="viridis",
    )
    ax0.scatter(obs_k_over_fwd, obs_t_days, c=obs_iv, cmap="viridis", s=12, edgecolors="white", linewidths=0.2)
    ax0.set_title(f"Predicted IV Heatmap ({day_label})")
    ax0.set_xlabel("K / Fwd")
    ax0.set_ylabel("T (days)")
    fig.colorbar(im, ax=ax0, label="Implied Vol")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.plot_surface(xx.numpy(), yy.numpy(), surf, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax1.scatter(obs_k_over_fwd, obs_t_days, obs_iv, color="black", s=8, depthshade=False)
    ax1.set_title(f"Predicted IV Surface ({day_label})")
    ax1.set_xlabel("K / Fwd")
    ax1.set_ylabel("T (days)")
    ax1.set_zlabel("Implied Vol")

    save_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path.resolve()


def main() -> None:
    l.seed_everything(42, workers=True)

    dm = SurfaceDataModule(
        data_path="./data/108105/2025_C_options_data.csv",
        num_workers=0,
    )
    dm.setup("fit")

    lit = LitSurfaceModel(
        spec=dm.spec,
        lr=2e-4,
        w_arb=1,
        w_smooth=1,
        vit_layers=12,
        vit_heads=12,
        d_model=768,
        mlp_size=3072
    )

    lrmon = LearningRateMonitor(logging_interval="epoch")
    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename="surface-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True,
    )
    trainer = l.Trainer(
        max_epochs=150,
        callbacks=[lrmon, ckpt],
        gradient_clip_val=1.0,
        enable_progress_bar=True,

    )

    trainer.fit(lit, dm)

    out = visualize_surface_for_one_day(lit, dm, out_path="surface_one_day.png", sample_index=0)
    print(f"Saved one-day surface visualization to {out}")


if __name__ == "__main__":
    main()
