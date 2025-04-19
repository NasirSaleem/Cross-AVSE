import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import AVSEDataModule
from model import AVSEModule


def configure_callbacks():
    return [
        ModelCheckpoint(
            monitor="val_loss",
            filename="best-{epoch:03d}-{val_loss:.3f}",
            save_top_k=3,
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]


def configure_trainer(args):
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=args.version
    )

    return Trainer(
        logger=logger,
        callbacks=configure_callbacks(),
        max_epochs=args.max_epochs,
        accelerator="gpu" if not args.cpu else "cpu",
        devices=args.gpus if not args.cpu else 1,
        precision="16-mixed" if args.fp16 else 32,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=args.deterministic,
        log_every_n_steps=args.log_every_n_steps,
        detect_anomaly=args.detect_anomaly,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        strategy=args.strategy,
        enable_progress_bar=not args.silent
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train AVSE Model")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Attention parameters
    parser.add_argument("--attention_layers", type=int, default=3)
    parser.add_argument("--attention_dim", type=int, default=256)

    # Data
    parser.add_argument("--lips", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment_name", type=str, default="avse")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    # Hardware
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--strategy", type=str, default=None)

    # Debugging
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--overfit_batches", type=float, default=0.0)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--silent", action="store_true")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    datamodule = AVSEDataModule(
        batch_size=args.batch_size,
        lips=args.lips,
        rgb=args.rgb,
        num_workers=args.num_workers
    )

    model = AVSEModule(
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_dataset=datamodule.val_dataset
    )

    trainer = configure_trainer(args)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
