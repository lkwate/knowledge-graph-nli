import pytorch_lightning as pl
import torch
from ..utils import *
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data, HeteroData
from typing import Any, Dict, Union, List
import pandas as pd
from loguru import logger
import os

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
        input_sentence = self.tokenizer(sentence1, sentence2, return_tensors="pt")
        
        graph_input1 = Data(x=graph1["pos_tag"].unsqueeze(-1), edge_index=graph1["edge_index"], edge_attr=graph1["edge_attr"])
        tokens1 = TokenList(tokens=graph1["tokens"])
        
        graph_input2 = Data(x=graph2["pos_tag"].unsqueeze(-1), edge_index=graph2["edge_index"], edge_attr=graph2["edge_attr"])
        tokens2 = TokenList(tokens=graph2["tokens"])
        
        return graph_input1, graph_input2, input_sentence, tokens1, tokens2, label


class MixedCollater(Collater):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        
    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, TokenList):
            output = []
            for item in batch:
                output.extend(item.tokens)
            return output
        else:
            return super().collate(batch)
        
class MixedDataLoader(DataLoader):
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
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str] = [],
        exclude_keys: List[str] = [],
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=MixedCollater(follow_batch,
                                             exclude_keys), **kwargs)
        
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
        return MixedDataLoader(
            self.train_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return MixedDataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return MixedDataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
