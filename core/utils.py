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


@dataclass
class TokenList:
    tokens: List[str]
