"""Definition of the Double Probing Transformer"""
from typing import Any, Dict, Optional
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import BertTokenizer
from .utils import create_sinusoidal_embeddings, get_masks, to_tensor, multi_acc, concat_batches
from .dataset import concat_before_return
from pytorch_lightning import LightningModule


class Encoder(BertModel):
    def __init__(self, config: BertConfig, sinusoidal_embeddings=False):
        super(Encoder, self).__init__(config, add_pooling_layer=False)
        if sinusoidal_embeddings:
            with torch.no_grad():  # RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
                create_sinusoidal_embeddings(
                    config.max_position_embeddings,
                    config.hidden_size,
                    out=self.embeddings.position_embeddings.weight,
                )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        x = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return x["last_hidden_state"]


class DpsaLayer(BertEncoder):
    def __init__(self, config : BertConfig):
        super(DpsaLayer, self).__init__(config)
        self.pooler = BertPooler(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        #return_dict=True,
    ) : 
        #bs, seq_len, hidden_size = hidden_states.shape
        #x = super()(
        x = super().forward(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            head_mask = head_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict=True
        )
        sequence_output = x["last_hidden_state"]
        pooler_output = self.pooler(sequence_output)
        return sequence_output, pooler_output

class DPTransformer(nn.Module):
    def __init__(
        self,
        attn: Encoder,
        dpsa: DpsaLayer,
        tokenizer: BertTokenizer,
        max_input_length: int = 512,
    ):
        super().__init__()
        self.attn = attn
        self.dpsa = dpsa
        self.tokenizer = tokenizer
        self.pad_index = tokenizer.pad_token_id
        self.max_input_length = max_input_length

    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.encode(sentence)
        return tokens

    def to_tensor(self, sentences):
        return to_tensor(
            sentences,
            pad_index=self.pad_index,
            tokenize=self.tokenize_and_cut,
            batch_first=True,
        )

    def dot_product(self, x):
        """x : (slen, dim) or (bs, slen, dim)"""
        if x.dim() == 2:
            return torch.matmul(x.t(), x)  # (dim, dim)
        elif x.dim() == 3:
            return torch.matmul(x.transpose(2, 1), x)  # (bs, dim, dim)
        else:
            raise RuntimeError("x.dim() != 2 and x.dim() != 3 is not supported")

    def positions(self, x):
        """x : (bs, slen)"""
        slen = x.size(1)
        positions = x.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)  # (bs, slen)
        return positions

    def forward(self, x, lengths_x, y, lengths_y, positions = None):
        """
        input : `x`, `y` (list of bs sentences indices) : (bs, x_len) and (bs, y_len)
        output : dpsa(x, y), dpsa(y, x)
        """
        bs = x.shape[0]

        x_len = x.size(1)
        y_len = y.size(1)
        assert lengths_x.size(0) == bs and lengths_y.size(0) == bs
        assert lengths_x.max().item() <= x_len and lengths_y.max().item() <= y_len

        # positions_id
        if positions is None:
            positions_x = self.positions(x)  # (bs, x_len)
            positions_y = self.positions(y)  # (bs, y_len)
        else:
            assert len(positions) == 2
            positions_x = positions[0]
            positions_y = positions[1]
            assert positions_x.size() == (bs, x_len)
            assert positions_y.size() == (bs, y_len)

        # token_type_ids
        token_type_ids_x = torch.zeros_like(x)
        token_type_ids_y = torch.zeros_like(y)

        # generate masks (padding_mask and attention mask)
        """
        - x_mask | y_mask : Mask to avoid performing attention on the padding token indices of the encoder input. 
        This mask is used in the cross-attention if the model is configured as a decoder. 
        Mask values selected in [0, 1]
        - x_attn_mask | y_attn_mask : Mask to avoid performing attention on padding token indices. 
        Mask values selected in [0, 1]
        - 1 for tokens that are not masked and 0 for tokens that are masked.
        """
        x_mask, x_attn_mask = get_masks(
            x_len, lengths_x, causal=False
        )  # (bs, x_len), (bs, x_len)
        y_mask, y_attn_mask = get_masks(
            y_len, lengths_y, causal=False
        )  # (bs, y_len), (bs, y_len)
        x_mask, x_attn_mask = x_mask.int(), x_attn_mask.int()
        y_mask, y_attn_mask = y_mask.int(), y_attn_mask.int()

        x = self.attn(
            input_ids=x,
            attention_mask=x_attn_mask,
            token_type_ids=token_type_ids_x,
            position_ids=positions_x,
            encoder_attention_mask=x_mask,
        )  # (bs, x_len, dim)

        y = self.attn(
            input_ids=y,
            attention_mask=y_attn_mask,
            token_type_ids=token_type_ids_y,
            position_ids=positions_y,
            encoder_attention_mask=y_mask,
        )  # (bs, y_len, dim)

        # TODO?? (batch_size, from_seq_length, to_seq_length)
        x_attn_mask = None  # (bs, x_len, y_len)
        y_attn_mask = None  # (bs, y_len, x_len)

        a_x_y = self.dpsa(
            hidden_states=x,
            attention_mask=x_attn_mask,
            encoder_hidden_states=y,
            encoder_attention_mask=x_mask,
        )  # (bs, x_len, dim)

        a_y_x = self.dpsa(
            hidden_states=y,
            attention_mask=y_attn_mask,
            encoder_hidden_states=x,
            encoder_attention_mask=y_mask,
        )  # (bs, y_len, dim)

        # a_x_y = self.dot_product(a_x_y)  # (bs, dim, dim)
        # a_y_x = self.dot_product(a_y_x)  # (bs, dim, dim)

        return a_x_y, a_y_x

    def forward_test(self, x, y):
        """
        The sentences must be of the same length (they must be completed in advance, with the padding token for example)
        """
        assert len(x) == len(y)
        # 'input_ids', 'token_type_ids', 'attention_mask'
        inputs_x = self.tokenizer(x, return_tensors="pt")
        inputs_y = self.tokenizer(y, return_tensors="pt")

        x = self.attn(**inputs_x)  # (bs, x_len, dim)
        y = self.attn(**inputs_y)  # (bs, y_len, dim)

        # TODO? (batch_size, from_seq_length, to_seq_length)
        x_attn_mask = None  # (bs, x_len, y_len)
        y_attn_mask = None  # (bs, y_len, x_len)

        a_x_y = self.dpsa(
            hidden_states=x,
            attention_mask=x_attn_mask,
            encoder_hidden_states=y,
            encoder_attention_mask=inputs_x["attention_mask"],
        )  # (bs, x_len, dim)

        a_y_x = self.dpsa(
            hidden_states=y,
            attention_mask=y_attn_mask,
            encoder_hidden_states=x,
            encoder_attention_mask=inputs_y["attention_mask"],
        )  # (bs, y_len, dim)

        # a_x_y = self.dot_product(a_x_y)  # (bs, dim, dim)
        # a_y_x = self.dot_product(a_y_x)  # (bs, dim, dim)

        return a_x_y, a_y_x

class LightningDPTransformer(LightningModule):
    def __init__(
        self,
        attn: Encoder,
        dpsa: DpsaLayer,
        tokenizer: BertTokenizer,
        config: Dict[str, Any],
        max_input_length: int = 512,
    ):
        super().__init__()
        self.model = DPTransformer(attn, dpsa, tokenizer, max_input_length)
        self.dropout = nn.Dropout(config["dropout"])
        self.inner_linear = nn.Linear(2 * config["hidden_dim"], config["hidden_dim"])
        self.outer_linear = nn.Linear(config["hidden_dim"], 1)
        self.lr = config["lr"]
        self.factor = config["lr_decay"]
        self.patience = config["lr_patience_scheduling"]
        self.criterion = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.factor, patience=self.patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return output

    def compute_loss(self, batch):
        x, y, label = batch[0], batch[1], batch[2]
        x, lengths_x = x[0].squeeze(1), x[1].squeeze(1)
        y, lengths_y = y[0].squeeze(1), y[1].squeeze(1)
        out_x_y, out_y_x = self.model(x, lengths_x, y, lengths_y)
        _, out_x_y = out_x_y
        _, out_y_x = out_y_x
        out = torch.cat([out_x_y, out_y_x], dim=-1)
        out = self.inner_linear(out)
        out = self.outer_linear(out).squeeze(-1)
        loss = self.criterion(out, label)
        acc = multi_acc(y_pred = out, y_test = label)
        return loss, acc

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.compute_loss(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.compute_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
    

class LightningBertMNLI(LightningModule):
    def __init__(
        self,
        attn: Encoder,
        tokenizer: BertTokenizer,
        config: Dict[str, Any],
        max_input_length: int = 512,
    ):
        super().__init__()
        self.attn = attn
        self.tokenizer = tokenizer
        self.pad_index = tokenizer.pad_token_id
        self.max_input_length = max_input_length
        
        self.dropout = nn.Dropout(config["dropout"])
        self.inner_linear = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.outer_linear = nn.Linear(config["hidden_dim"], 3)
        self.lr = config["lr"]
        self.factor = config["lr_decay"]
        self.patience = config["lr_patience_scheduling"]
        self.criterion = nn.CrossEntropyLoss()

    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.encode(sentence)
        return tokens

    def positions(self, x):
        """x : (bs, slen)"""
        slen = x.size(1)
        positions = x.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)  # (bs, slen)
        return positions

    def forward(self, x, lengths, segment_ids = None, positions = None):
        """
        input : `x` (list of bs sentences indices) : (bs, len)
        """
        bs = x.shape[0]
        
        len_ = x.size(1)
        assert lengths.size(0) == bs
        assert lengths.max().item() <= len_ 

        # positions_id
        if positions is None:
            positions = self.positions(x)  # (bs, len)
        else:
            assert positions.size() == (bs, len_)

        # token_type_ids
        token_type_ids_x = segment_ids if segment_ids is not None else torch.zeros_like(x)

        # generate masks (padding_mask and attention mask)
        """
        - mask : Mask to avoid performing attention on the padding token indices of the encoder input. 
        This mask is used in the cross-attention if the model is configured as a decoder. 
        Mask values selected in [0, 1]
        - attn_mask : Mask to avoid performing attention on padding token indices. 
        Mask values selected in [0, 1]
        - 1 for tokens that are not masked and 0 for tokens that are masked.
        """
        mask, attn_mask = get_masks(
            len_, lengths, causal=False
        )  # (bs, len), (bs, len)
        mask, attn_mask = mask.int(), attn_mask.int()

        x = self.attn(
            input_ids=x,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids_x,
            position_ids=positions,
            encoder_attention_mask=mask,
        )  # (bs, len, dim)
        
        # [CLS] token
        x = x[:,0] # (bs, dim)
        
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.factor, patience=self.patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return output

    def compute_loss(self, batch):
        if concat_before_return :
            (x, lengths, positions, segment_ids), label = batch
            x, lengths = x[0].squeeze(1), x[1].squeeze(1)
        else :
            x, y, label = batch[0], batch[1], batch[2]
            x1, len1 = x[0].squeeze(1), x[1].squeeze(1)
            x2, len2 = y[0].squeeze(1), y[1].squeeze(1)
            #x1, len1 = x[0], x[1]
            #x2, len2 = y[0], y[1]
            x, lengths, positions, segment_ids = concat_batches(x1, len1, x2, len2, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id)
        
        out = self(x, lengths, segment_ids, positions)
        #out = self.__call__(x, lengths, segment_ids, positions)
        out = self.inner_linear(out)
        out = self.outer_linear(out).squeeze(-1)
        loss = self.criterion(out, label.long())
        acc = multi_acc(y_pred = out, y_test = label)
        return loss, acc

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.compute_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss