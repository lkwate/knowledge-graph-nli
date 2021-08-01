"""Utiliy functions for the building of the core of the model"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, BertLayer
from transformers.models.bert.modeling_bert import BertEncoder
import numpy as np
import os
import copy


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    """Create sinusoidal embedding for
    Args:
        n_pos: size of one sinusoidal encoded vector
        dim: dimension on which the positional encoding will be applied
        out: output tensor"""
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


def to_tensor(
    sentences, pad_index, dico=None, tokenize=None, batch_first=False, max_length=512
):
    if type(sentences) == str:
        sentences = [sentences]
    else:
        assert type(sentences) in [list, tuple]

    assert (dico is None) ^ (tokenize is None)

    if dico is not None:
        # Here the tokens of the sentence must be separated by the blank word
        sentences = [s.strip().split() for s in sentences]
        bs = len(sentences)
        lengths = [len(sent) for sent in sentences]
        slen = max(max_length, max(lengths))
        lengths = torch.LongTensor(lengths)
        word_ids = torch.LongTensor(slen, bs).fill_(pad_index)
        for i in range(bs):
            sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
            word_ids[: len(sent), i] = sent
        if batch_first:
            return word_ids.transpose(0, 1), lengths
        else:
            return word_ids, lengths

    else:
        sentences = [tokenize(s) for s in sentences]
        bs = len(sentences)
        lengths = [len(sent) for sent in sentences]
        slen = max(max_length, max(lengths))
        lengths = torch.LongTensor(lengths)
        word_ids = torch.LongTensor(bs, slen).fill_(pad_index)
        for i in range(bs):
            sent = torch.LongTensor(sentences[i])
            word_ids[i, : len(sent)] = sent
        if batch_first:
            return word_ids, lengths
        else:
            return word_ids.transpose(0, 1), lengths
