"""Utiliy functions for the building of the core of the model"""

import torch
import numpy as np
import os
import hashlib
import ntpath

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
    Concat batches.
    x1, len1 : premise 
    x2, len2 : hypothesis
    """
    lengths = len1 + len2 + 3 # [CLS] premise [SEP] hypothesis [SEP]
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(bs,  slen).fill_(pad_token_id) 
    x[:,0].fill_(cls_token_id) # [CLS] 

    l = len1.max().item()
    x[:,1:l+1].copy_(x1[:,:l]) # [CLS] premise 

    segment_ids = torch.LongTensor(bs, slen).fill_(pad_token_id)
    for i in range(bs):
        l1, l2 = len1[i], len2[i]
        l3 = l1+l2+2
        x[i,l1+1] = sep_token_id # [CLS] premise [SEP] 
        x[i,l1+2:l3].copy_(x2[i,:l2]) # [CLS] premise [SEP] hypothesis
        x[i,l3] = sep_token_id # [CLS] premise [SEP] hypothesis [SEP]

        #segment_ids[i,:] = torch.tensor([0] * (l1 + 2) + [1] * (l2 + 1) + [0] * (slen - l3 - 1))  # sentence 0 & sentence 1 & pad_index
        segment_ids[i,:l3+1] = torch.tensor([0] * (l1 + 2) + [1] * (l2 + 1)) # sentence 0 & sentence 1 

    assert (x == cls_token_id).long().sum().item() ==  bs
    assert (x == sep_token_id).long().sum().item() == 2 * bs

    positions = torch.arange(slen)[:, None].repeat(1, bs).to(x1.device).t()

    return x, lengths, positions, segment_ids.to(x1.device)

def path_leaf(path):
    # https://stackoverflow.com/a/8384788/11814682
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_hash_object(type_='sha-1'):
    """make a hash object"""
    assert type_ in ["sha-1", "sha-256", "md5"]
    if type_ == 'sha-1' :
        h = hashlib.sha1()
    elif type_ == "sha-256":
        h = hashlib.sha256()
    elif type_ == "md5" :
        h = hashlib.md5()
    return h

def hash_file(file_path, BLOCK_SIZE = 65536, type_='sha-1'):
    """This function returns the SHA-1/SHA-256/md5 hash of the file passed into it
    #  BLOCK_SIZE : the size of each read from the file
    https://www.programiz.com/python-programming/examples/hash-file
    https://nitratine.net/blog/post/how-to-hash-files-in-python/
    """
    assert os.path.isfile(file_path)
    # make a hash object
    h = get_hash_object(type_)
    # open file for reading in binary mode
    with open(file_path,'rb') as file:
        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only BLOCK_SIZE bytes at a time
            chunk = file.read(BLOCK_SIZE)
            h.update(chunk)
    # return the hex representation of digest #, hash value as a bytes object
    return h.hexdigest() #, h.digest()

def hash_var(var, type_='sha-1'):
    """This function returns the SHA-1/SHA-256/md5 hash of the variable passed into it
    https://nitratine.net/blog/post/how-to-hash-files-in-python/
    https://stackoverflow.com/questions/24905062/how-to-hash-a-variable-in-python"""
    # make a hash object
    h = get_hash_object(type_)
    h.update(var.encode('utf8'))
    # return the hex representation of digest #, hash value as a bytes object
    return h.hexdigest() #, h.digest()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_data_path(config, data_file, n_samples) :
    filename, _ = os.path.splitext(path_leaf(data_file))
    params = AttrDict(config)
    #f = '%s_%s_%s_%s_%s'%(params.batch_size, params.max_length, params.in_memory, n_samples, params.dpsa)
    f = '%s_%s_%s_%s'%(params.max_length, params.in_memory, n_samples, params.dpsa)
    filename = "%s_%s"%(filename, hash_var(f))
    data_path = os.path.join(params.dump_path, '%s.pth'%filename)
    return data_path