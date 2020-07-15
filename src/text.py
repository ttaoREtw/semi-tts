"""Modified from tensorflow_datasets.features.text.*

Reference: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text_lib
"""
import abc
import numpy as np
import pandas as pd
from os.path import basename

SEP = '\t'

class _BaseTextEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self):
        raise NotImplementedError

    @abc.abstractproperty
    def vocab_size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def token_type(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load_from_file(cls, vocab_file):
        raise NotImplementedError

    @property
    def pad_idx(self):
        return 0

    @property
    def space_idx(self):
        return 1
    @property
    def eos_idx(self):
        return 2

    # @property
    # def unk_idx(self):
    #     return -1

    def __repr__(self):
        return "<{} vocab_size={}>".format(type(self).__name__, self.vocab_size)


class PhoneTextEncoder(_BaseTextEncoder):
    def __init__(self, vocab_list):
        # Note that vocab_list must not contain <pad>, <space>
        # <pad>=0, <space>=1
        self._vocab_list = ["<pad>", "<space>", "<eos>"] + vocab_list # remove "<eos>"/"<unk>"
        self._vocab2idx = {v: np.uint8(idx) for idx, v in enumerate(self._vocab_list)}
        self.map_table = None

    def encode(self, s):
        # Always strip trailing space, \r and \n, split by space
        s = s.strip("\r\n ").split(' ')
        # Manually append eos to the end
        # with punctuation version:
        return [self.vocab_to_idx(v) if v != '' else self.space_idx for v in s] + [self.pad_idx]
        # without punctuation version:
        #return [self.vocab_to_idx(v) for v in s if v != ''] + [self.eos_idx]
        #return [self.vocab_to_idx(v) for v in s if v != ''] + [self.pad_idx]


    def decode(self, ids):
        vocabs = []
        for i in ids:
            v = self.idx_to_vocab(i)
            #if v == "<pad>":
            #    vocabs.append('<pad>')
            #elif v == "<eos>":
            #    vocabs.append(v) # Include <eos>
            #else:
            vocabs.append(v)
        return " ".join(vocabs)

    @classmethod
    def load_from_file(cls, vocab_file):
        with open(vocab_file, "r") as f:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in f]
        return cls(vocab_list)

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def token_type(self):
        return 'phoneme'

    def vocab_to_idx(self, vocab):
        # return self._vocab2idx.get(vocab, self.unk_idx)
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

    def set_map_table(self, table_path):
        # Setup table of phn sequence of particular wave files
        self.map_table = pd.read_csv(table_path, index_col=0, sep=SEP)

    def file_to_seq(self, file_path):
        file_id = basename(file_path).split('.')[0] # file_id = file name w/o type
        file_phn_seq = self.map_table.loc[file_id].phn_seq
        return self.encode(file_phn_seq)

    def file_to_spkr(self, file_path):
        file_id = basename(file_path).split('.')[0] # file_id = file name w/o type
        spkr = self.map_table.loc[file_id].spkr
        return spkr


def load_text_encoder(mode, vocab_file, map_table=None):
    if mode == "phoneme":
        text_encoder = PhoneTextEncoder.load_from_file(vocab_file)
    else:
        raise NotImplementedError("`{}` is not yet supported.".format(mode))

    if map_table is not None:
        text_encoder.set_map_table(map_table)

    return text_encoder

