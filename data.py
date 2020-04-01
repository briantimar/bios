import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
from collections import Counter
from datetime import datetime

class StringDataset:
    """Wrapper with integer indices around the json database. 
    The whole thing is loaded into memory. Does not impose order on entries."""

    def __init__(self, fname):
        """fname = path to db file."""
        self.fname = fname
        with open(self.fname) as f:
            _db = json.load(f)
            self.names = list(_db.keys())
            self.bios = [_db[n] for n in self.names]
        
    def __len__(self):
        return len(self.bios)

    def __getitem__(self, i):
        return self.bios[i]


class ByteWrapper:
    """Yields bytestrings terminated with a STOP value"""
    
    STOP_CODE = 205
    START_CODE = 204

    def __init__(self, string_dataset):
        """fname = path to db file."""
        self.string_dataset = string_dataset
        self._get_byte_values()

    def _get_byte_values(self, fname="byte_values.txt"):
        """Returns dict mapping bytes to consecutive integer indices."""
        with open(fname) as f:
            bytes_list = f.readline().split(',')
        bytes_list = [int(b.strip()) for b in bytes_list]
        assert len(bytes_list) == self.num_codes -2
        self._bytes_list = bytes_list
        self._byte_value_map = {bytes_list[i]: i for i in range(len(bytes_list))}

    def __len__(self):
        return len(self.string_dataset)
    
    def _to_int_seq(self, s):
        """Given a string s, returns corresponding list of integer codes and appends a STOP
        code on the end."""
        return [self.START_CODE] + [self._byte_value_map[b] for b in s.encode()] + [self.STOP_CODE]

    def _to_string(self, int_seq):
        """Given int seq terminated in STOP, return decoded string."""
        bts = [self._bytes_list[i] for i in int_seq[1:-1]]
        return bytes(bts).decode(errors="replace")

    def __getitem__(self, i):
        return self._to_int_seq(self.string_dataset[i])

    @property
    def num_codes(self):
        """number of byte codes including start and stop."""
        return self.STOP_CODE + 1

class ByteDataset(Dataset):

    def __init__(self, fname, one_hot=True):
        """fname = path to json db
            one_hot: whether to yield byte values as one-hot"""
        super().__init__()
        self.fname = fname
        self.one_hot = one_hot
        self._strds = StringDataset(fname)
        self._bytecodes = ByteWrapper(self._strds)
    
    def __getitem__(self, i):
        bts = self._bytecodes[i]
        if self.one_hot:
            t = torch.zeros(len(bts), self._bytecodes.num_codes, dtype=torch.float)
            t[range(len(bts)), bts] = 1
        else:
            t = torch.tensor(self._bytecodes[i], dtype=torch.long)
        return t
    
    def __len__(self):
        return len(self._bytecodes)

    def string(self, bytecodes):
        """Returns string corresponding to a list of bytecodes."""
        return self._bytecodes._to_string(bytecodes)

class ByteDataLoader(DataLoader):
    """Loads packed sequences of integer codes for each batch."""

    def __init__(self, byte_ds, **kwargs):
        if not byte_ds.one_hot:
            raise ValueError
        super().__init__(byte_ds, collate_fn=lambda t: pack_sequence(t, enforce_sorted=False), 
                                **kwargs)
    


if __name__ == "__main__":
    bds = ByteDataset("bios.json")
    bytcodes = bds[0]
