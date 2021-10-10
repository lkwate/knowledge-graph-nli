import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geo_nn
import pytorch_lightning as pl
import torch
from .utils import *
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Tuple, Union, List
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from loguru import logger
import click
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
DATASET_TYPE = {"train", "val", "test"}


class GraphDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: Dict[str, Union[str, float, int]],
        type: str,
    ) -> None:
        super().__init__()
        if type not in DATASET_TYPE:
            err_msg = "Unknown dataset type. The right dataset type falls into the following categories: 'train', 'val', 'test'"
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.tokenizer = tokenizer

        if type == "train":
            logger.info("Train data loading")
            self.data = pd.read_csv(config["train_data_path"])
        elif type == "val":
            logger.info("Validation data loading")
            self.data = pd.read_csv(config["val_data_path"])
        else:
            logger.info("Testing data loading")
            self.data = pd.read_csv(config["test_data_path"])

        self.label_factory = {"neutral": 0, "entailment": 1, "contradiction": 2}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx]
        sentence1, sentence2, label = (
            item["sentence1"],
            item["sentence2"],
            item["label"],
        )
        label = float(self.label_factory[label])

        graph_sentence1 = dependency_tree(sentence1)
        graph_sentence2 = dependency_tree(sentence2)
        input_sentence = self.tokenizer(sentence1, sentence2, return_tensors="pt")

        return graph_sentence1, graph_sentence2, input_sentence, label


class GraphModel(nn.Module):

    criterion = nn.CrossEntropyLoss()

    def __init__(self, config: Dict[str, Any]):
        super(GraphModel, self).__init__()
        self.pos_num = config["pos_num"]
        self.pos_dim = config["pos_dim"]
        self.edge_num = config["edge_num"]
        self.edge_dim = config["edge_dim"]
        self.dropout = config.get("dropout", 0.1)
        self.embedding_dim = config.get("embedding_dim", 768)
        self.num_transformer_conv_head = config.get("num_transformer_conv_head", 4)
        self.num_transformer_conv = config.get("num_transformer_conv", 4)
        self.num_class = config.get("num_class", 3)
        self.transformer_conv_dim = self.pos_dim + 768
        self.model_name = config.get("model_name", "bert-base-uncased")

        self.bert = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pos_embedding = nn.Embedding(self.pos_num, self.pos_dim)
        self.edge_embedding = nn.Embedding(self.edge_num, self.edge_dim)
        self.graph_transformer = nn.Sequential(
            *[
                geo_nn.TransformerConv(
                    self.transformer_conv_dim,
                    self.transformer_conv_dim,
                    heads=self.num_transformer_conv_head,
                    dropout=self.dropout,
                    edge_dim=self.edge_dim,
                    concat=False,
                )
            ]
            * self.num_transformer_conv
        )

        self.graph_merging = nn.Linear(2 * self.transformer_conv_dim, 768)
        self.outer_projection = nn.Linear(768 * 2, self.num_class)
        self.dropout = nn.Dropout(self.dropout)

    def _forward_graph_transformer(self, graph_model, graph_input):
        edge_attr = graph_input["edge_attr"]
        edge_index = graph_input["edge_index"]
        x = graph_input["x"]
        for layer in graph_model:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = torch.mean(x, dim=-2)
        return x

    def forward(self, graph_input1, graph_input2, transformer_input):
        transformer_input, graph_input1, graph_input2 = self._pre_embedding(
            transformer_input, graph_input1, graph_input2
        )
        out_bert = self.bert(**transformer_input)["pooler_output"]
        out_bert = self.dropout(out_bert)
        out_graph1 = self._forward_graph_transformer(
            self.graph_transformer, graph_input1
        )
        out_graph2 = self._forward_graph_transformer(
            self.graph_transformer, graph_input2
        )
        out_graph = torch.cat([out_graph1, out_graph2], dim=-1)
        out_graph = self.graph_merging(out_graph).unsqueeze(0)
        out = torch.cat([out_bert, out_graph], dim=-1)
        out = self.dropout(out)
        out = self.outer_projection(out)
        return out

    def compute_loss(self, graph_input1, graph_input2, transformer_input, label):
        out = self.forward(graph_input1, graph_input2, transformer_input)
        loss = self.criterion(out, label.long())
        return loss

    def _sub_embedding(self, tokens: List[Tuple[str]], dummy_tensor: torch.Tensor):
        tokens = map(lambda item: item[0], tokens)
        embeddings = []
        for token in tokens:
            sub_tokens = self.tokenizer(
                token, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            sub_tokens = sub_tokens.type_as(dummy_tensor).long()
            sub_embedding = self.bert.embeddings.word_embeddings(sub_tokens)[0]
            sub_embedding = torch.max(sub_embedding, dim=0).values
            embeddings.append(sub_embedding)

        embeddings = torch.stack(embeddings)
        return embeddings

    def _pre_embedding(self, transformer_input, graph_input1, graph_input2):
        transformer_input = {k: v.squeeze(1) for k, v in transformer_input.items()}
        tokens1 = graph_input1["tokens"]
        tokens2 = graph_input2["tokens"]
        del graph_input1["tokens"], graph_input2["tokens"]
        graph_input1 = {k: v.squeeze(0) for k, v in graph_input1.items()}
        graph_input2 = {k: v.squeeze(0) for k, v in graph_input2.items()}

        graph_input1["edge_attr"] = self.edge_embedding(graph_input1["edge_attr"])
        graph_input2["edge_attr"] = self.edge_embedding(graph_input2["edge_attr"])
        dummy_tensor = graph_input1["edge_attr"]
        graph_input1["x"] = torch.cat(
            [
                self._sub_embedding(tokens1, dummy_tensor),
                self.pos_embedding(graph_input1["pos_tag"]),
            ],
            dim=-1,
        )
        graph_input2["x"] = torch.cat(
            [
                self._sub_embedding(tokens2, dummy_tensor),
                self.pos_embedding(graph_input2["pos_tag"]),
            ],
            dim=-1,
        )
        del graph_input1["pos_tag"], graph_input2["pos_tag"]

        return transformer_input, graph_input1, graph_input2


class GraphLightningDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        if "train_data_path" not in config:
            err_msg = (
                "train_data_path not found in the dataset configurations dictionary"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        if "val_data_path" not in config:
            err_msg = "val_data_path not found in the dataset configurations dictionary"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if "test_data_path" not in config:
            err_msg = "test_data_path not found in the dataset configuration dictionary"
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        logger.info("Train dataset...")
        self.train_dataset = GraphDataset(self.tokenizer, config, "train")
        logger.info("Validation dataset...")
        self.val_dataset = GraphDataset(self.tokenizer, config, "val")
        logger.info("Testing dataset...")
        self.test_dataset = GraphDataset(self.tokenizer, config, "test")
        self.num_workers = config.get("num_workers", os.cpu_count())

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class GraphLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = GraphModel(config)
        self.lr = config["lr"]
        self.factor = config["lr_decay"]
        self.patience = config["lr_patience_scheduling"]
        self.accumulate_grad_batches = config["accumulate_grad_batches"]
        self.skip_step = 0
        self.training_step_outputs = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.factor, patience=self.patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return output

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch, batch_idx)
        output = {"loss": loss, "train_accuracy": accuracy}

        self.training_step_outputs.append([loss.detach().cpu(), accuracy])
        if (self.skip_step + 1) % self.accumulate_grad_batches == 0:
            train_loss = (
                torch.stack(list(map(lambda item: item[0], self.training_step_outputs)))
                .float()
                .mean()
            )
            train_accuracy = (
                torch.stack(list(map(lambda item: item[1], self.training_step_outputs)))
                .float()
                .mean()
            )
            out_log = {"train_loss": train_loss, "train_accuracy": train_accuracy}
            self.skip_step = 0
            self.training_step_outputs = []

            self.log_dict(out_log, prog_bar=True)

        self.skip_step += 1
        return output

    def _metric_forward(self, batch, batch_idx):
        graph_input1, graph_input2, transformer_input, label = self._unpack_batch(batch)
        out = self.model(graph_input1, graph_input2, transformer_input)
        loss = self.model.criterion(out, label.long())
        predicted_class = torch.argmax(out, dim=-1).item()
        acc = int(label.long().item() == predicted_class)

        return loss, torch.tensor(acc)

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch, batch_idx)
        output = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(output)

        return output

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch, batch_idx)
        output = {"test_loss": loss, "test_accuracy": accuracy}
        self.log_dict(output)

        return output

    def validation_epoch_end(self, outputs):
        out = torch.stack(list(map(lambda item: item["val_accuracy"], outputs))).float()
        accuracy = torch.mean(out).item()

        self.log("final_val_accuracy", accuracy, prog_bar=True)

    def _unpack_batch(self, batch):
        return batch[0], batch[1], batch[2], batch[3]


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("val_data_path", type=click.Path(exists=True))
@click.argument("test_data_path", type=click.Path(exists=True))
@click.argument("batch_size", type=int)
@click.option("--model_name", type=str, default="roberta-base")
@click.option("--lr", type=float, default=1e-5)
@click.option("--lr_decay", type=float, default=0.8)
@click.option("--lr_patience_scheduling", type=int, default=3)
@click.option("--max_epochs", type=int, default=1)
@click.option("--val_check_interval", type=float, default=0.20)
@click.option("--patience_early_stopping", type=float, default=5)
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--dropout", type=float, default=0.1)
@click.option("--pos_dim", type=int, default=16)
@click.option("--edge_dim", type=int, default=16)
@click.option("--num_transformer_conv_head", type=int, default=4)
@click.option("--num_transformer_conv", type=int, default=4)
@click.option("--num_class", type=int, default=3)
@click.option("--seed", type=int, default=42)
@click.option("--save_top_k", type=int, default=5)
def train(
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    batch_size: int,
    model_name: str,
    lr: float,
    lr_decay: float,
    lr_patience_scheduling: int,
    max_epochs: int,
    val_check_interval: float,
    patience_early_stopping: int,
    accumulate_grad_batches: int,
    dropout: float,
    edge_dim: int,
    pos_dim: int,
    num_transformer_conv_head: int,
    num_class: int,
    num_transformer_conv: int,
    seed: int,
    save_top_k: int,
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
        "edge_dim": edge_dim,
        "pos_dim": pos_dim,
        "num_transformer_conv_head": num_transformer_conv_head,
        "pos_num": len(POS_DICT) + 1,
        "edge_num": len(DEP_DICT) + 1,
        "num_class": num_class,
        "num_transformer_conv": num_transformer_conv,
        "seed": seed,
    }

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    logger.info("Model initialisation...")
    model = GraphLightningModule(config)

    logger.info("Data Module initialisation...")
    datamodule = GraphLightningDataModule(config)

    config_trainer = {
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
        filename="{epoch}-{final_val_accuracy:.3f}",
        monitor="final_val_accuracy",
        save_weights_only=True,
        save_top_k=save_top_k,
    )
    config_trainer["callbacks"] = [early_stopping_callback]
    config_trainer["callbacks"].append(model_checkpoint_callback)
    trainer = pl.Trainer(**config_trainer)
    trainer.fit(model, datamodule)

    logger.info("Testing")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
