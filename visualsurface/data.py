from __future__ import annotations

from typing import List, Optional, Tuple

import lightning as l
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from visualsurface.math_ops import rasterize_quotes
from visualsurface.types import RasterSpec, SurfaceBatch


def make_quote_numeric_features(
    S: torch.Tensor,
    K: torch.Tensor,
    T_years: torch.Tensor,
    price: torch.Tensor,
    bid: torch.Tensor,
    ask: torch.Tensor,
    delta: torch.Tensor,
    gamma: torch.Tensor,
    theta: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    vix: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-12
    mid = 0.5 * (bid + ask)
    spread = torch.clamp(ask - bid, min=0.0)
    spot = torch.clamp(S, min=eps)

    gamma_scaled = gamma * (spot ** 2)
    theta_scaled = theta * torch.clamp(T_years, min=1e-6)
    logK_over_S = torch.log(torch.clamp(K, min=eps) / spot)

    return torch.stack(
        [
            mid / spot,
            spread / spot,
            price / spot,
            delta,
            gamma_scaled,
            theta_scaled,
            r,
            q,
            logK_over_S,
            vix,
        ],
        dim=-1,
    )


class DayGroupedDataset(Dataset):
    def __init__(self, grouped_rows: List[dict]):
        self.rows = grouped_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        return self.rows[item]


class SurfaceDataModule(l.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        Nu: int = 64,
        Nv: int = 32,
        patch: int = 4,
        only_calls: bool = True,
        only_eur: bool = True,
        train_ratio: float = 0.9,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        u_quantile_clip: Tuple[float, float] = (0.001, 0.999),
        v_quantile_clip: Tuple[float, float] = (0.001, 0.999),
    ):
        super().__init__()
        self.data_path = data_path
        self.Nu = Nu
        self.Nv = Nv
        self.patch = patch
        self.only_calls = only_calls
        self.only_euro = only_eur
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.u_quantile_clip = u_quantile_clip
        self.v_quantile_clip = v_quantile_clip

        self.spec: Optional[RasterSpec] = None
        self.train_ds: Optional[DayGroupedDataset] = None
        self.val_ds: Optional[DayGroupedDataset] = None

        self.quote_num_mean: Optional[torch.Tensor] = None
        self.quote_num_std: Optional[torch.Tensor] = None

        self.feat_ix = {
            "Impl_Vol": 0,
            "Bid": 1,
            "Ask": 2,
            "delta": 3,
            "gamma": 4,
            "theta": 5,
        }

    def _read_df(self) -> pl.DataFrame:
        if self.data_path.endswith(".parquet"):
            return pl.read_parquet(self.data_path)
        return pl.read_csv(self.data_path)

    def _compute_quote_num_stats(self, df_train: pl.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        expr_spot = pl.col("S").clip(lower_bound=eps)
        expr_mid = 0.5 * (pl.col("Bid") + pl.col("Ask"))
        expr_spread = (pl.col("Ask") - pl.col("Bid")).clip(lower_bound=0.0)
        expr_gamma_scaled = pl.col("gamma") * (expr_spot ** 2)
        expr_theta_scaled = pl.col("theta") * pl.col("T_years").clip(lower_bound=1e-6)
        expr_logK_over_S = (pl.col("K") / expr_spot).log()

        feat_exprs = [
            (expr_mid / expr_spot).alias("f0"),
            (expr_spread / expr_spot).alias("f1"),
            (pl.col("Price") / expr_spot).alias("f2"),
            pl.col("delta").alias("f3"),
            expr_gamma_scaled.alias("f4"),
            expr_theta_scaled.alias("f5"),
            pl.col("rate").alias("f6"),
            pl.col("dividend_yield").alias("f7"),
            expr_logK_over_S.alias("f8"),
            pl.col("vix").alias("f9"),
        ]

        tmp = df_train.select(feat_exprs)
        means = tmp.select([pl.col(f"f{i}").mean().alias(f"m{i}") for i in range(10)]).to_dicts()[0]
        stds = tmp.select([pl.col(f"f{i}").std().fill_null(1.0).alias(f"s{i}") for i in range(10)]).to_dicts()[0]

        mean = torch.tensor([float(means[f"m{i}"]) for i in range(10)], dtype=torch.float32)
        std = torch.tensor([float(stds[f"s{i}"]) for i in range(10)], dtype=torch.float32)
        std = torch.clamp(std, min=1e-6)
        return mean, std

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("date").str.to_date().alias("date"),
            pl.col("S").cast(pl.Float64),
            pl.col("K").cast(pl.Float64),
            pl.col("T").cast(pl.Int32),
            pl.col("vix").cast(pl.Float64),
            pl.col("Bid").cast(pl.Float64),
            pl.col("Ask").cast(pl.Float64),
            pl.col("Price").cast(pl.Float64),
            pl.col("Impl_Vol").cast(pl.Float64),
            pl.col("dividend_yield").cast(pl.Float64),
            pl.col("rate").cast(pl.Float64),
            pl.col("delta").cast(pl.Float64),
            pl.col("gamma").cast(pl.Float64),
            pl.col("theta").cast(pl.Float64),
        )

        df = df.filter(
            (pl.col("T") > 0)
            & (pl.col("T") < 1000)
            & (pl.col("S") > 0)
            & (pl.col("K") > 0)
            & (pl.col("Impl_Vol") > 0)
            & (pl.col("Ask") >= pl.col("Bid"))
        )

        if self.only_calls:
            df = df.filter(pl.col("cp_flag") == "C")
        if self.only_euro:
            df = df.filter(pl.col("exercise_style") == "E")

        df = df.with_columns((pl.col("T").cast(pl.Float64) / 365.0).alias("T_years"))
        df = df.with_columns(
            (pl.col("S") * ((pl.col("rate") - pl.col("dividend_yield")) * pl.col("T_years")).exp()).alias("Fwd")
        )
        df = df.with_columns((pl.col("K") / pl.col("Fwd")).log().alias("u"))
        df = df.with_columns(pl.col("T_years").log().alias("v"))

        df = df.with_columns(
            pl.when(pl.col("cp_flag") == "C").then(0).otherwise(1).cast(pl.Int64).alias("cp_i"),
            pl.when(pl.col("exercise_style") == "E").then(0).otherwise(1).cast(pl.Int64).alias("style_i"),
        )

        return df

    def _compute_spec_from_train(self, df_train: pl.DataFrame) -> RasterSpec:
        uq0, uq1 = self.u_quantile_clip
        vq0, vq1 = self.v_quantile_clip

        stats = df_train.select(
            pl.col("u").quantile(uq0).alias("u_lo"),
            pl.col("u").quantile(uq1).alias("u_hi"),
            pl.col("v").quantile(vq0).alias("v_lo"),
            pl.col("v").quantile(vq1).alias("v_hi"),
        ).to_dicts()[0]

        u_min = float(stats["u_lo"]) - 0.3
        u_max = float(stats["u_hi"]) + 0.3
        v_min = float(stats["v_lo"]) - 0.20
        v_max = float(stats["v_hi"]) + 0.20

        assert self.Nu % self.patch == 0 and self.Nv % self.patch == 0, "Nu/Nv must be divisible by patch"

        return RasterSpec(
            Nu=self.Nu,
            Nv=self.Nv,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
        )

    def _group_by_date(self, df: pl.DataFrame) -> List[dict]:
        grouped = (
            df.group_by("date")
            .agg(
                pl.col("S").first().alias("S"),
                pl.col("vix").first().alias("vix"),
                pl.col("rate").median().alias("r_med"),
                pl.col("dividend_yield").median().alias("q_med"),
                pl.col("u").alias("u_list"),
                pl.col("v").alias("v_list"),
                pl.col("K").alias("K_list"),
                pl.col("T").alias("T_days_list"),
                pl.col("Bid").alias("Bid_list"),
                pl.col("Ask").alias("Ask_list"),
                pl.col("Price").alias("Price_list"),
                pl.col("Impl_Vol").alias("IV_list"),
                pl.col("rate").alias("r_list"),
                pl.col("dividend_yield").alias("q_list"),
                pl.col("delta").alias("delta_list"),
                pl.col("gamma").alias("gamma_list"),
                pl.col("theta").alias("theta_list"),
                pl.col("cp_i").alias("cp_list"),
                pl.col("style_i").alias("style_list"),
            )
            .sort("date")
        )
        return grouped.to_dicts()

    def setup(self, stage: Optional[str] = None):
        df = self._read_df()
        df = self._preprocess(df)

        dates = df.select(pl.col("date").unique().sort()).to_series().to_list()
        n = len(dates)
        n_train = max(1, int(self.train_ratio * n))
        print(f"N_Dates: {n} | N_Dates_Train: {n_train}")
        train_dates = set(dates[:n_train])
        val_dates = set(dates[n_train:]) if n_train < n else set(dates[-1:])

        df_train = df.filter(pl.col("date").is_in(list(train_dates)))
        df_val = df.filter(pl.col("date").is_in(list(val_dates)))

        self.spec = self._compute_spec_from_train(df_train)
        self.quote_num_mean, self.quote_num_std = self._compute_quote_num_stats(df_train)

        self.train_ds = DayGroupedDataset(self._group_by_date(df_train))
        self.val_ds = DayGroupedDataset(self._group_by_date(df_val))

    def collate_fn(self, batch_rows: List[dict]) -> SurfaceBatch:
        assert self.spec is not None
        assert self.quote_num_mean is not None and self.quote_num_std is not None

        B = len(batch_rows)
        Ns = [len(r["u_list"]) for r in batch_rows]
        maxN = max(Ns)

        def pad_1d(vals: List[float], pad_value: float = 0.0) -> torch.Tensor:
            t = torch.tensor(vals, dtype=torch.float32)
            if t.numel() < maxN:
                t = F.pad(t, (0, maxN - t.numel()), value=pad_value)
            return t

        def pad_1d_long(vals: List[int], pad_value: int = 0) -> torch.Tensor:
            t = torch.tensor(vals, dtype=torch.long)
            if t.numel() < maxN:
                t = F.pad(t, (0, maxN - t.numel()), value=pad_value)
            return t

        quote_u = torch.stack([pad_1d(r["u_list"]) for r in batch_rows], dim=0)
        quote_v = torch.stack([pad_1d(r["v_list"]) for r in batch_rows], dim=0)
        K = torch.stack([pad_1d(r["K_list"]) for r in batch_rows], dim=0)
        T_days = torch.stack([pad_1d(r["T_days_list"]) for r in batch_rows], dim=0)
        bid = torch.stack([pad_1d(r["Bid_list"]) for r in batch_rows], dim=0)
        ask = torch.stack([pad_1d(r["Ask_list"]) for r in batch_rows], dim=0)
        price = torch.stack([pad_1d(r["Price_list"]) for r in batch_rows], dim=0)
        quote_iv = torch.stack([pad_1d(r["IV_list"]) for r in batch_rows], dim=0)
        r_q = torch.stack([pad_1d(r["r_list"]) for r in batch_rows], dim=0)
        q_q = torch.stack([pad_1d(r["q_list"]) for r in batch_rows], dim=0)
        delta = torch.stack([pad_1d(r["delta_list"]) for r in batch_rows], dim=0)
        gamma = torch.stack([pad_1d(r["gamma_list"]) for r in batch_rows], dim=0)
        theta = torch.stack([pad_1d(r["theta_list"]) for r in batch_rows], dim=0)
        cp = torch.stack([pad_1d_long(r["cp_list"]) for r in batch_rows], dim=0)
        style = torch.stack([pad_1d_long(r["style_list"]) for r in batch_rows], dim=0)

        quote_valid = torch.zeros(B, maxN, dtype=torch.bool)
        for i, n in enumerate(Ns):
            quote_valid[i, :n] = True

        spot = torch.tensor([float(r["S"]) for r in batch_rows], dtype=torch.float32)
        vix = torch.tensor([float(r["vix"]) for r in batch_rows], dtype=torch.float32)
        r_med = torch.tensor([float(r["r_med"]) for r in batch_rows], dtype=torch.float32)
        q_med = torch.tensor([float(r["q_med"]) for r in batch_rows], dtype=torch.float32)
        global_feats = torch.stack([torch.log(torch.clamp(spot, min=1e-12)), vix, r_med, q_med], dim=-1)

        T_years = T_days / 365.0
        S_bn = spot.view(B, 1).expand(B, maxN)
        vix_bn = vix.view(B, 1).expand(B, maxN)

        quote_num = make_quote_numeric_features(
            S=S_bn,
            K=K,
            T_years=T_years,
            price=price,
            bid=bid,
            ask=ask,
            delta=delta,
            gamma=gamma,
            theta=theta,
            r=r_q,
            q=q_q,
            vix=vix_bn,
        )

        mean = self.quote_num_mean.view(1, 1, -1)
        std = self.quote_num_std.view(1, 1, -1)
        quote_num = (quote_num - mean) / std

        feat = torch.stack([quote_iv, bid, ask, delta, gamma, theta], dim=-1)
        img = rasterize_quotes(quote_u, quote_v, feat, quote_valid, self.spec, self.feat_ix)

        batch = SurfaceBatch(
            img=img,
            quote_u=quote_u,
            quote_v=quote_v,
            quote_num=quote_num,
            cp=cp,
            style=style,
            quote_valid=quote_valid,
            global_feats=global_feats,
            quote_iv=quote_iv,
            K=K,
            T_days=T_days,
            r_q=r_q,
            q_q=q_q,
            spot=spot,
        )
        batch.validate(self.spec)
        return batch

    def train_dataloader(self):
        assert self.train_ds is not None
        use_mps = torch.backends.mps.is_available()
        num_workers = 0 if use_mps else self.num_workers
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
            persistent_workers=(num_workers > 0),
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        use_mps = torch.backends.mps.is_available()
        num_workers = 0 if use_mps else self.num_workers
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
            persistent_workers=(num_workers > 0),
        )
