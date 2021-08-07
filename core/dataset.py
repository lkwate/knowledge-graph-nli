"""Dataset definition"""
from core.utils import to_tensor
from typing import Dict, Union, List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
from transformers import AutoTokenizer
import pandas as pd
from loguru import logger


class NLIDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: Dict[str, Union[str, float, int]],
        train: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if train:
            self.data = pd.read_csv(config["train_data_path"])
        else:
            self.data = pd.read_csv(config["val_data_path"])
        self.data = self.data[self.data["label"] != "neutral"]
        self.max_length = config.get("max_length", 512)
        self.label_factory = {"entailment": 1, "contradiction": -1, "neutral": 0}

    def sentence_and_cut(self, sentence):
        try:
            tokens = self.tokenizer.tokenize(sentence)
        except:
            print(sentence)
        tokens = tokens[: self.max_length]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def to_tensor(self, sentences):
        return to_tensor(
            sentences,
            pad_index=self.tokenizer.pad_token_id,
            tokenize=self.sentence_and_cut,
            batch_first=True,
        )

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx]
        output1 = self.to_tensor(item["sentence1"])
        output2 = self.to_tensor(item["sentence2"])
        label = item["label"]
        label = self.label_factory[label]

        return output1, output2, label


class NLIDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: Dict[str, Union[str, float, int]],
    ):
        super().__init__()

        if "train_data_path" not in config:
            logger.error(
                "train_data_path not found in the dataset configurations dictionary"
            )
            raise ValueError(
                "train_data_path not found in the dataset configurations dictionary"
            )

        if "val_data_path" not in config:
            logger.error(
                "val_data_path not found in the dataset configurations dictionary"
            )
            raise ValueError(
                "val_data_path not found in the dataset configurations dictionary"
            )

        self.train_dataset = NLIDataset(tokenizer, config, train=True)
        self.val_dataset = NLIDataset(tokenizer, config, train=False)
        self.num_workers = config.get("num_workers", 2)
        self.batch_size = config["batch_size"]

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
