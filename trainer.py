import copy
from loguru import logger
import torch
from core.dataset import NLIDataModule
from core.model import Encoder, DpsaLayer, LightningDPTransformer
import click
from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
from core.model import LightningDPTransformer, LightningBertMNLI
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
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--train_n_samples", type=int, default=-1)
@click.option("--val_n_samples", type=int, default=-1)
@click.option("--dropout", type=float, default=0.1)
@click.option("--dpsa", type=bool, default=True)
@click.option('--in_memory', type=bool, default=True)
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
    hidden_dim: int,
    accumulate_grad_batches: int,
    train_n_samples: int,
    val_n_samples: int,
    dropout: float,
    dpsa : bool,
    in_memory : bool,
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
        "hidden_dim": hidden_dim,
        "train_n_samples": train_n_samples,
        "val_n_samples": val_n_samples,
        "dropout" : dropout,
        "dpsa" : dpsa,
        "in_memory" : in_memory
    }
    
    # logger.info("")
    logger.info(f"model initialisation from {model_name}")
    attn_pretrained = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    attn = Encoder(attn_pretrained.config)
    state_dict = {
        key: value
        for key, value in attn_pretrained.state_dict().items()
        if not key.startswith("pooler")
    }
    attn.load_state_dict(state_dict)
        
    if dpsa :
        dpsa_pretrained = copy.deepcopy(attn_pretrained.encoder)
        dpsa = DpsaLayer(dpsa_pretrained.config)
        pooler_state_dict = attn_pretrained.pooler.state_dict()
        pooler_state_dict["pooler.dense.weight"] = pooler_state_dict["dense.weight"]
        pooler_state_dict["pooler.dense.bias"] = pooler_state_dict["dense.bias"]
        del pooler_state_dict["dense.weight"], pooler_state_dict["dense.bias"]
        dpsa.load_state_dict({**attn_pretrained.encoder.state_dict(), **pooler_state_dict})

        model = LightningDPTransformer(attn, dpsa, tokenizer, config, max_length)
    else : # Bert + classification layer
        model = LightningBertMNLI(attn, tokenizer, config, max_length)
        
    logger.info("Dataset...")
    data_module = NLIDataModule(tokenizer, config)

    logger.info("model initialisation completed")
    config_trainer = {
        "max_epochs": max_epochs,
        "val_check_interval": val_check_interval,
        "accumulate_grad_batches": accumulate_grad_batches,
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
