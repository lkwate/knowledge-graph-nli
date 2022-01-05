import pytorch_lightning as pl
import torch
from ..utils import *
from transformers import RobertaTokenizer, PreTrainedTokenizerBase
from torch.utils.data import Dataset
from torch_geometric.loader.dataloader import Collater, DataLoader
from torch_geometric.data import Data, HeteroData
from typing import Any, Dict, Union, List
import pandas as pd
from loguru import logger
import os

DATASET_TYPE = {"train", "val", "test"}


class GraphDataset(Dataset):
    def __init__(
        self,
        tokenizer: RobertaTokenizer,
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

        self.add_global_token = config["add_global_token"]
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

        graph1 = dependency_tree(
            sentence1, self.tokenizer, add_global_token=self.add_global_token
        )
        graph2 = dependency_tree(
            sentence2, self.tokenizer, add_global_token=self.add_global_token
        )
        transformer_input = self.tokenizer(sentence1, sentence2)

        graph_input1 = Data(
            x=graph1["pos_tag"].unsqueeze(-1),
            edge_index=graph1["edge_index"],
            edge_attr=graph1["edge_attr"],
        )
        tokens1 = TokenList(tokens=graph1["tokens"])

        graph_input2 = Data(
            x=graph2["pos_tag"].unsqueeze(-1),
            edge_index=graph2["edge_index"],
            edge_attr=graph2["edge_attr"],
        )
        tokens2 = TokenList(tokens=graph2["tokens"])

        output = {
            "graph_input1": graph_input1,
            "graph_input2": graph_input2,
            "transformer_input": transformer_input,
            "tokens1": tokens1,
            "tokens2": tokens2,
            "label": label,
        }

        return output


class MixedCollater(Collater):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, follow_batch, exclude_keys):
        self.tokenizer = tokenizer
        self.torch_geometric_collator = Collater(follow_batch, exclude_keys)


    def _collate_token_list(self, batch):
        output = []
        for item in batch:
            output.extend(item.tokens)
        return output

    def collate(self, batch):
        tokens1, tokens2, transformer_input = [], [], []
        for feat in batch:
            tokens1.append(feat["tokens1"])
            tokens2.append(feat["tokens2"])
            transformer_input.append(feat["transformer_input"])
            del feat["tokens1"], feat["tokens2"], feat["transformer_input"]

        tokens1 = self._collate_token_list(tokens1)
        tokens2 = self._collate_token_list(tokens2)
        transformer_input = self.tokenizer.pad(transformer_input, return_tensors="pt")
        
        batch = self.torch_geometric_collator.collate(batch)
        batch["tokens1"] = tokens1
        batch["tokens2"] = tokens2
        batch["transformer_input"] = transformer_input
        
        return batch


class MixedDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`[]`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int,
        tokenizer: PreTrainedTokenizerBase,
        shuffle: bool = False,
        follow_batch: List[str] = [],
        exclude_keys: List[str] = [],
        **kwargs,
    ):
        self.tokenizer = tokenizer
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=MixedCollater(self.tokenizer, follow_batch, exclude_keys),
            **kwargs,
        )


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

        self.batch_size = self.config["batch_size"]
        self.tokenizer = RobertaTokenizer.from_pretrained(config["model_name"])
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
        return MixedDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return MixedDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return MixedDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            num_workers=self.num_workers,
        )
