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

def multi_acc(y_pred, y_test):
    """
    y_pred : logits of size  (bs, n_class) or predicted class of size (bs,)
    y_test : expected class of size (bs,)
    """
    if y_pred.dim() == 2 :
        y_pred = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
    acc = (y_pred == y_test).sum().float() / float(y_test.size(0))
    return acc.item()

def concat_batches(x1, len1, x2, len2, cls_token_id, sep_token_id, pad_token_id):
    """
    Concat batches with different languages.
    """
    lengths = len1 + len2 + 3 # [CLS] premise [SEP] hypothesis [SEP]
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(bs,  slen).fill_(pad_token_id)
    x[:,0].fill_(cls_token_id)

    l = len1.max().item()
    x[:,1:l+1].copy_(x1[:,:l])

    segment_ids = torch.LongTensor(bs, slen).fill_(pad_token_id)
    for i in range(bs):
        l1, l2 = len1[i], len2[i]
        l3 = l1+l2+2
        x[i,l1+1] = sep_token_id
        x[i,l1+2:l3].copy_(x2[i,:l2])
        x[i,l3] = sep_token_id

        #segment_ids[i,:] = torch.tensor([0] * (l1 + 2) + [1] * (l2 + 1) + [0] * (slen - l3 - 1))  # sentence 0 & sentence 1 & pad_index
        segment_ids[i,:l3+1] = torch.tensor([0] * (l1 + 2) + [1] * (l2 + 1)) # sentence 0 & sentence 1 

    assert (x == cls_token_id).long().sum().item() ==  bs
    assert (x == sep_token_id).long().sum().item() == 2 * bs

    positions = torch.arange(slen)[:, None].repeat(1, bs).to(x1.device).t()

    return x, lengths, positions, segment_ids.to(x1.device)