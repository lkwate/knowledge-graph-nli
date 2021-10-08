import copy
import os
from loguru import logger
import torch
from core.dataset import NLIDataModule
from core.model import Encoder, DpsaLayer, LightningDPTransformer
import click
from transformers import BertModel, BertTokenizer
from core.model import LightningDPTransformer, LightningBertMNLI

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("val_data_path", type=click.Path(exists=True))
@click.argument("test_data_path", type=click.Path(exists=True))
@click.argument("batch_size", type=int)
@click.option("--model_name", type=str, default="bert-base-uncased")
@click.option("--lr", type=float, default=1e-5)
@click.option("--lr_decay", type=float, default=0.8)
@click.option("--lr_patience_scheduling", type=int, default=3)
@click.option("--max_length", type=int, default=512)
@click.option("--max_epochs", type=int, default=10)
@click.option("--val_check_interval", type=float, default=0.5)
@click.option("--patience_early_stopping", type=float, default=5)
@click.option("--validation_metrics", type=str, default="val_acc", help="Validation metrics : val_acc, val_loss ...")
@click.option("--hidden_dim", type=int, default=768)
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--train_n_samples", type=int, default=-1)
@click.option("--val_n_samples", type=int, default=-1)
@click.option("--test_n_samples", type=int, default=-1)
@click.option("--dropout", type=float, default=0.1)
@click.option("--dpsa", type=bool, default=True)
@click.option('--in_memory', type=bool, default=True)
@click.option("--random_seed", type=int, default=2021)
@click.option("--dump_path", type=str, default=None, help="Experiment dump path")
@click.option("--exp_id", type=str, default="text_entailment", help="Experiment ID")
@click.option("--reload_checkpoint", type=str, default="", help="Reload a checkpoint")
@click.option("--eval_only", type=bool, default=False, help="Only run evaluations")
@click.option("--auto_scale_batch_size", type=str, default=None, # "binsearch" 
            help="Automatically tries to find the largest batch size that fits into memory, before any training")
@click.option("--auto_lr_find", type=bool, default=False, help="runs a learning rate finder algorithm")
@click.option("--deterministic", type=bool, default=False, help='ensures reproducibility')
def main(
    train_data_path: str,
    val_data_path: str,
    test_data_path : str,
    batch_size: int,
    lr: float,
    lr_decay: float,
    lr_patience_scheduling: int,
    max_length: int,
    model_name: str,
    max_epochs: int,
    val_check_interval: float,
    patience_early_stopping: float,
    validation_metrics : str,
    hidden_dim: int,
    accumulate_grad_batches: int,
    train_n_samples: int,
    val_n_samples: int,
    test_n_samples : int,
    dropout: float,
    dpsa : bool,
    in_memory : bool,
    random_seed : int,
    dump_path : int,
    exp_id : int,
    reload_checkpoint : str,
    eval_only : bool,
    auto_scale_batch_size : str,
    auto_lr_find : bool,
    deterministic : bool
):
    config = {
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "test_data_path":test_data_path,
        "batch_size": batch_size,
        "lr": lr,
        "lr_decay": lr_decay,
        "lr_patience_scheduling": lr_patience_scheduling,
        "max_length": max_length,
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "train_n_samples": train_n_samples,
        "val_n_samples": val_n_samples,
        "test_n_samples" : test_n_samples,
        "dropout" : dropout,
        "dpsa" : dpsa,
        "in_memory" : in_memory,
        "eval_only" : eval_only,
        "dump_path" : dump_path
    }
    
    resume_from_checkpoint = reload_checkpoint if os.path.isfile(reload_checkpoint) else None
    assert not eval_only or os.path.isfile(resume_from_checkpoint if resume_from_checkpoint else "")
    pl.seed_everything(random_seed, workers=True)
    root_dir = os.path.join(dump_path, exp_id)
    if not os.path.isdir(dump_path) :
        os.mkdir(dump_path)
    if not os.path.isdir(root_dir) :
        os.mkdir(root_dir)

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
        
        "default_root_dir" : root_dir,
        #"log_every_n_steps" : max(len(data_module.train_dataset) // batch_size, 0),
        "resume_from_checkpoint" : resume_from_checkpoint,
        #"weights_save_path" : dump_path,
        "auto_scale_batch_size":auto_scale_batch_size, # None
        "auto_select_gpus" : True,
        "auto_lr_find":auto_lr_find,
        "benchmark" : False,
        "deterministic" : deterministic
    }
    if not eval_only :
        config_trainer["log_every_n_steps"] = max(len(data_module.train_dataset) // batch_size, 0)
        
    if torch.cuda.is_available():
        config_trainer["gpus"] = -1

    mode = "min" if 'loss' in validation_metrics else 'max'
    early_stopping_callback = EarlyStopping(
        monitor=validation_metrics, patience=patience_early_stopping, verbose=False, strict=True,
        mode = mode
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor=validation_metrics,
        dirpath=root_dir,
        filename="model-{epoch:02d}-{%s:.2f}"%validation_metrics,
        save_top_k=3,
        mode=mode,
    )
    
    config_trainer["callbacks"] = [early_stopping_callback, checkpoint_callback, PrintCallback()]
    trainer = pl.Trainer(**config_trainer)
    
    if not eval_only :
        trainer.fit(model, data_module)
        trainer.test(dataloaders=data_module.test_dataloader())
    else :
        if dpsa :
            model = LightningDPTransformer.load_from_checkpoint(resume_from_checkpoint, attn = attn, dpsa = dpsa, tokenizer = tokenizer, config = config, max_input_length = max_length)
        else :
            model = LightningBertMNLI.load_from_checkpoint(resume_from_checkpoint, attn = attn, tokenizer = tokenizer, config = config, max_input_length = max_length)
        trainer.test(model, dataloaders=data_module.test_dataloader())
        
class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("=========== Training is started! ========")
    def on_train_end(self, trainer, pl_module):
        print("======== Training is done. ======== ")

if __name__ == "__main__":
    main()
