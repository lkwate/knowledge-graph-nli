import pytorch_lightning as pl
import torch
from ..utils import *
from transformers import AutoConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from .model import GraphLightningModule
from .dataset import GraphLightningDataModule
from loguru import logger
import click
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("val_data_path", type=click.Path(exists=True))
@click.argument("test_data_path", type=click.Path(exists=True))
@click.argument("batch_size", type=int)
@click.option("--checkpoint_path", type=click.Path(exists=True))
@click.option("--model_name", type=str, default="lkwate/roberta-base-mnli")
@click.option("--lr", type=float, default=1e-5)
@click.option("--lr_decay", type=float, default=0.8)
@click.option("--lr_patience_scheduling", type=int, default=3)
@click.option("--max_epochs", type=int, default=1)
@click.option("--val_check_interval", type=float, default=0.20)
@click.option("--patience_early_stopping", type=float, default=5)
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--dropout", type=float, default=0.1)
@click.option("--num_transformer_conv_head", type=int, default=4)
@click.option("--num_transformer_conv", type=int, default=6)
@click.option("--num_class", type=int, default=3)
@click.option("--seed", type=int, default=42)
@click.option("--save_top_k", type=int, default=5)
@click.option("--add_global_token", is_flag=True)
@click.option("--freeze_bert", is_flag=True)
@click.option("--log_path", type=click.Path())
@click.option("--embedding_dim", type=int, default=256)
@click.option("--optimizer_name", type=str, default="Lamb")
@click.option("--action", type=str, default="train")
def main(
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    batch_size: int,
    checkpoint_path: str,
    model_name: str,
    lr: float,
    lr_decay: float,
    lr_patience_scheduling: int,
    max_epochs: int,
    val_check_interval: float,
    patience_early_stopping: int,
    accumulate_grad_batches: int,
    dropout: float,
    num_transformer_conv_head: int,
    num_class: int,
    num_transformer_conv: int,
    seed: int,
    save_top_k: int,
    add_global_token: bool,
    log_path: str,
    embedding_dim: int,
    optimizer_name: str,
    action: str,
    freeze_bert: bool,
):
    config = {
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "test_data_path": test_data_path,
        "batch_size": batch_size,
        "model_name": model_name,
        "lr": lr,
        "lr_decay": lr_decay,
        "lr_patience_scheduling": lr_patience_scheduling,
        "max_epochs": max_epochs,
        "val_check_interval": val_check_interval,
        "patience_early_stopping": patience_early_stopping,
        "accumulate_grad_batches": accumulate_grad_batches,
        "dropout": dropout,
        "num_transformer_conv_head": num_transformer_conv_head,
        "pos_num": len(POS_DICT) + 1,
        "edge_num": len(DEP_DICT) + 1,
        "num_class": num_class,
        "num_transformer_conv": num_transformer_conv,
        "seed": seed,
        "add_global_token": add_global_token,
        "hidden_size": AutoConfig.from_pretrained(model_name).hidden_size,
        "embedding_dim": embedding_dim,
        "optimizer_name": optimizer_name,
        "freeze_bert": freeze_bert
    }

    torch.manual_seed(seed)
    logger.info(f"Model ({model_name}) initialisation...")
    if config["add_global_token"]:
        logger.info("CLS global token used in Graph Transformer")

    logger.info(f"Optimizer : {optimizer_name}")
    if checkpoint_path:
        logger.info(f"Load the model from checkpoint ...{checkpoint_path[-50:]}")
        model = GraphLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path, config=config
        )
    else:
        logger.info("Load the model from huggingface's pretrained")
        model = GraphLightningModule(config)

    logger.info("Data Module initialisation...")
    datamodule = GraphLightningDataModule(config)

    config_trainer = {
        "default_root_dir": log_path,
        "max_epochs": max_epochs,
        "val_check_interval": val_check_interval,
        "accumulate_grad_batches": max(accumulate_grad_batches, batch_size),
    }
    if torch.cuda.is_available():
        config_trainer["gpus"] = -1

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=patience_early_stopping, verbose=False, strict=True
    )
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        save_weights_only=True,
        save_top_k=save_top_k,
    )
    config_trainer["callbacks"] = [early_stopping_callback]
    config_trainer["callbacks"].append(model_checkpoint_callback)
    trainer = pl.Trainer(**config_trainer)

    if action == "train":
        logger.info("training...")
        trainer.fit(model, datamodule)

        logger.info("Testing")
        trainer.test(model=model, datamodule=datamodule)

    if action == "test":
        if not checkpoint_path:
            logger.error(
                "Checkpoint path should be defined to test a model a the starting of the script"
            )
        logger.info("Testing")
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
