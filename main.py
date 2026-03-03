import lightning as l
from lightning.pytorch.callbacks import LearningRateMonitor

from visualsurface import LitSurfaceModel, SurfaceDataModule

l.seed_everything(42, workers=True)

dm = SurfaceDataModule(
    data_path="./data/108105/2025_C_options_data.csv",
    num_workers=0,
)

dm.setup("fit")

lit = LitSurfaceModel(
    spec=dm.spec,
    lr=2e-4,
)

lrmon = LearningRateMonitor(logging_interval="epoch")
trainer = l.Trainer(
    max_epochs=10,
    callbacks=[lrmon],
    gradient_clip_val=1.0,
    enable_progress_bar=True,
)

trainer.fit(lit, dm)
