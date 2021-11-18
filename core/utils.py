"""Utiliy functions for the building of the core of the model"""

import torch
import numpy as np
import spacy
from collections import defaultdict
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List

nlp = spacy.load("en_core_web_sm")

POS_DICT = defaultdict(int)
DEP_DICT = defaultdict(int)

for i, pos in enumerate(
    [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CONJ",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
        "EOL",
        "SPACE",
    ]
):
    POS_DICT[pos] = i + 1

# Do not include the root edge in the dependency graph
for i, dep in enumerate(
    [
        "acl",
        "acomp",
        "advcl",
        "advmod",
        "agent",
        "amod",
        "appos",
        "attr",
        "aux",
        "auxpass",
        "case",
        "cc",
        "ccomp",
        "clf",
        "complm",
        "compound",
        "conj",
        "cop",
        "csubj",
        "csubjpass",
        "dative",
        "dep",
        "det",
        "discourse",
        "dislocated",
        "dobj",
        "expl",
        "fixed",
        "flat",
        "goeswith",
        "hmod",
        "hyph",
        "infmod",
        "intj",
        "iobj",
        "list",
        "mark",
        "meta",
        "neg",
        "nmod",
        "nn",
        "npadvmod",
        "nsubj",
        "nsubjpass",
        "nounmod",
        "npmod",
        "num",
        "number",
        "nummod",
        "oprd",
        "obj",
        "obl",
        "orphan",
        "parataxis",
        "partmod",
        "pcomp",
        "pobj",
        "poss",
        "possessive",
        "preconj",
        "prep",
        "prt",
        "punct",
        "quantmod",
        "rcmod",
        "relcl",
        "reparandum",
        "root",
        "vocative",
        "xcomp",
    ]
):
    DEP_DICT[dep] = i + 1


def dependency_tree(text: str, tokenizer: AutoTokenizer, add_global_token: bool = True):
    doc = nlp(text)
    pos_vec = []
    tokens = []

    edge_start_index = []
    edge_end_index = []
    edge_attr = []
    for token in doc:
        tokens.append(token.text)
        pos_vec.append(POS_DICT[token.pos_])
        dependency = DEP_DICT[token.dep_]

        if dependency:
            edge_start_index.append(token.head.i)
            edge_end_index.append(token.i)
            edge_attr.append(dependency)

    if add_global_token:
        tokens.append(tokenizer.cls_token)
        pos_vec.append(0)

        for token in doc:
            edge_start_index.append(token.i)
            edge_end_index.append(len(doc))
            edge_attr.append(0)

    pos_vec = torch.LongTensor(pos_vec)
    edge_index = torch.LongTensor([edge_start_index, edge_end_index])
    edge_attr = torch.LongTensor(edge_attr)

    output = {
        "pos_tag": pos_vec,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "tokens": tokens,
    }
    return output


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


@dataclass
class TokenList:
    tokens: List[str]
