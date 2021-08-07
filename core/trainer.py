import copy
from loguru import logger
import torch
from .dataset import NLIDataModule
from .model import Encoder, DpsaLayer, LightningDPTransformer
import click
from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
from .model import LightningDPTransformer
from pytorch_lightning.callbacks import EarlyStopping


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("val_data_path", type=click.Path(exists=True))
@click.argument("batch_size", type=int)
@click.option("--model_name", type=str, default="bert-base-uncased")
@click.option("--lr", type=float, default=1e-5)
@click.option("--lr_decay", type=float, default=0.8)
@click.option("--lr_patience_scheduling", type=int, default=3)
@click.option("--max_length", type=int, default=512)
@click.option("--max_epochs", type=int, default=10)
@click.option("--val_check_interval", type=float, default=0.5)
@click.option("--patience_early_stopping", type=float, default=5)
@click.option("--hidden_dim", type=int, default=768)
def main(
    train_data_path: str,
    val_data_path: str,
    batch_size: int,
    lr: float,
    lr_decay: float,
    lr_patience_scheduling: int,
    max_length: int,
    model_name: str,
    max_epochs: int,
    val_check_interval: float,
    patience_early_stopping: float,
    hidden_dim: int
):
    config = {
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "batch_size": batch_size,
        "lr": lr,
        "lr_decay": lr_decay,
        "lr_patience_scheduling": lr_patience_scheduling,
        "max_length": max_length,
        "model_name": model_name,
        "hidden_dim": hidden_dim
    }

    logger.info(f"model initialisation from {model_name}")
    att_pretrained = BertModel.from_pretrained(model_name)
    attn = Encoder(att_pretrained.config)
    state_dict = {
        key: value
        for key, value in att_pretrained.state_dict().items()
        if not key.startswith("pooler")
    }
    attn.load_state_dict(state_dict)

    dpsa_pretrained = copy.deepcopy(att_pretrained)
    dpsa = DpsaLayer(dpsa_pretrained.config)
    dpsa.load_state_dict(dpsa_pretrained.state_dict())

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = LightningDPTransformer(attn, dpsa, tokenizer, max_length)
    data_module = NLIDataModule(tokenizer, config)

    logger.info("model initialisation completed")
    config_trainer = {
        "max_epochs": max_epochs,
        "val_check_interval": val_check_interval,
    }
    if torch.cuda.is_available():
        config_trainer["gpus"] = -1

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=patience_early_stopping, verbose=False, strict=True
    )
    config_trainer["callbacks"] = [early_stopping_callback]
    trainer = pl.Trainer(**config_trainer)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
