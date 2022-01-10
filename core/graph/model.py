import torch.nn as nn
import torch.optim as optim
import torch_optimizer
import torch_geometric.nn as geo_nn
import pytorch_lightning as pl
import torch
from ..utils import *
from transformers import AutoModel, RobertaTokenizer
from typing import Any, Dict, Tuple, List
from loguru import logger

OPTIMIZER_DIC = {"Adam": optim.Adam, "Lamb": torch_optimizer.Lamb}


class GraphModel(nn.Module):

    criterion = nn.CrossEntropyLoss()

    def __init__(self, config: Dict[str, Any]):
        super(GraphModel, self).__init__()
        self.pos_num = config["pos_num"]
        self.edge_num = config["edge_num"]
        self.dropout = config.get("dropout", 0.1)
        self.embedding_dim = config["embedding_dim"]
        self.num_transformer_conv_head = config.get("num_transformer_conv_head", 12)
        self.num_transformer_conv = config.get("num_transformer_conv", 12)
        self.num_class = config.get("num_class", 3)
        self.model_name = config.get("model_name", "bert-base-uncased")
        self.add_global_token = config["add_global_token"]
        self.hidden_size = config["hidden_size"]
        self.freeze_bert = config.get("freeze_bert")

        self.bert = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.pos_embedding = nn.Embedding(self.pos_num, self.embedding_dim)
        self.edge_embedding = nn.Embedding(self.edge_num, self.embedding_dim)
        self.graph_transformer = nn.Sequential(
            *[
                geo_nn.TransformerConv(
                    self.embedding_dim + self.hidden_size,
                    self.embedding_dim + self.hidden_size,
                    heads=self.num_transformer_conv_head,
                    dropout=self.dropout,
                    edge_dim=self.embedding_dim,
                    concat=False,
                )
            ]
            * self.num_transformer_conv
        )

        self.graph_merging = nn.Linear(
            3 * (self.embedding_dim + self.hidden_size), self.hidden_size
        )
        self.outer_projection = nn.Linear(self.hidden_size * 2, self.num_class)
        self.dropout = nn.Dropout(self.dropout)

        if self.add_global_token:
            self.graph_aggregator = lambda x: x[-1, :]
        else:
            self.graph_aggregator = lambda x: torch.mean(x, dim=0)
    
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.required_grad = False
        
    def _forward_graph_transformer(self, graph_model, graph_input):
        edge_attr = graph_input.edge_attr
        edge_index = graph_input.edge_index
        x = graph_input.x
        ptr = graph_input.ptr.long()

        for layer in graph_model:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        out = [
            self.graph_aggregator(x[ptr[i] : ptr[i + 1], :])
            for i in range(ptr.shape[0] - 1)
        ]
        out = torch.stack(out)

        return out

    def forward(self, graph_input1, graph_input2, transformer_input, tokens1, tokens2):
        graph_input1 = self._pre_embedding(graph_input1, tokens1)
        graph_input2 = self._pre_embedding(graph_input2, tokens2)

        out_bert = self.bert(**transformer_input)["pooler_output"]
        out_bert = self.dropout(out_bert)
        out_graph1 = self._forward_graph_transformer(
            self.graph_transformer, graph_input1
        )
        out_graph2 = self._forward_graph_transformer(
            self.graph_transformer, graph_input2
        )
        out_graph = self._graph_merging_func(out_graph1, out_graph2)
        out = torch.cat([out_bert, out_graph], dim=-1)
        out = self.dropout(out)
        out = self.outer_projection(out)
        return out

    def _graph_merging_func(self, input1, input2):
        input1 = self.dropout(input1)
        input2 = self.dropout(input2)
        out_graph = torch.cat(
            [input1 + input2, input1 - input2, input1 * input2], dim=-1
        )
        out_graph = self.graph_merging(out_graph)

        return out_graph

    def compute_loss(self, graph_input1, graph_input2, transformer_input, label):
        out = self.forward(graph_input1, graph_input2, transformer_input)
        loss = self.criterion(out, label.long())
        return loss

    def _sub_embedding(self, tokens: List[Tuple[str]], device: torch.device):
        embeddings = []
        for token in tokens:
            sub_tokens = self.tokenizer(
                token, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            sub_tokens = sub_tokens.to(device)
            sub_embedding = self.bert.embeddings.word_embeddings(sub_tokens)[0]
            sub_embedding = torch.mean(sub_embedding, dim=0)
            embeddings.append(sub_embedding)

        embeddings = torch.stack(embeddings)
        return embeddings

    def _pre_embedding(self, graph_input, tokens):
        graph_input.x = graph_input.x.squeeze(-1)
        graph_input.edge_attr = graph_input.edge_attr.squeeze(-1)
        graph_input.edge_attr = self.edge_embedding(graph_input.edge_attr)
        device = graph_input.x.device
        graph_input.x = torch.cat(
            [
                self._sub_embedding(tokens, device),
                self.pos_embedding(graph_input.x),
            ],
            dim=-1,
        )

        return graph_input


class GraphLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = GraphModel(config)
        self.optimizer_name = config["optimizer_name"]
        self.lr = config["lr"]
        self.factor = config["lr_decay"]
        self.patience = config["lr_patience_scheduling"]
        self.accumulate_grad_batches = config["accumulate_grad_batches"]
        self.skip_step = 0
        self.training_step_outputs = []

    def configure_optimizers(self):
        optimizer = OPTIMIZER_DIC[self.optimizer_name](
            self.model.parameters(), lr=self.lr
        )
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
        self.log_dict(output)

        return output

    def _metric_forward(self, batch, batch_idx):
        graph_input1, graph_input2, transformer_input, tokens1, tokens2, label = (
            batch["graph_input1"],
            batch["graph_input2"],
            batch["transformer_input"],
            batch["tokens1"],
            batch["tokens2"],
            batch["label"],
        )
        out = self.model(
            graph_input1, graph_input2, transformer_input, tokens1, tokens2
        )
        loss = self.model.criterion(out, label.long())
        predicted_class = torch.argmax(out, dim=-1)
        acc = (label.long() == predicted_class).float().mean()

        return loss, acc

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
