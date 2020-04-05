import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
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

class ByteCode:
    """Wrapper around byte codes: converts from integer codes to strings."""

   
    MISSING = b' '[0]

    def __init__(self, fname):
        """fname = path to byte code csv."""
        self.fname = fname
        self.STOP_CODE = None
        self._get_byte_values(self.fname)
    
    def _get_byte_values(self, fname):
        """Returns dict mapping bytes to consecutive integer indices."""
        with open(fname) as f:
            bytes_list = f.readline().split(',')
        bytes_list = [int(b.strip()) for b in bytes_list]
        self.STOP_CODE = len(bytes_list)
        self._bytes_list = bytes_list
        self._byte_value_map = {bytes_list[i]: i for i in range(len(bytes_list))}

    def _to_code(self, b):
        """returns code for a given byte value.
            If not in byte value map, returns code of MISSING"""
        if b not in self._byte_value_map:
            return self._byte_value_map[self.MISSING]
        return self._byte_value_map[b]

    def to_int_seq(self, s):
        """Given a string s, returns corresponding list of integer codes and appends a STOP
        code on the end."""
        return [self.STOP_CODE] + [self._to_code(b) for b in s.encode()] + [self.STOP_CODE]

    def to_string(self, int_seq):
        """Given int seq terminated in STOP, return decoded string."""
        bts = [self._bytes_list[i] for i in int_seq[1:-1]]
        return bytes(bts).decode(errors="replace")

    @property
    def num_codes(self):
        """number of byte codes including stop."""
        return self.STOP_CODE + 1

class ByteWrapper:
    """Yields bytestrings terminated with a STOP value"""
    
    def __init__(self, string_dataset, byte_code):
        """string_dataset = instance of StringDataset
        byte_code = instance of ByteCode"""
        self.string_dataset = string_dataset
        self.byte_code = byte_code

    def __len__(self):
        return len(self.string_dataset)
    
    def _to_int_seq(self, s):
        """Given a string s, returns corresponding list of integer codes and appends a STOP
        code on the end."""
        return self.byte_code.to_int_seq(s)

    def _to_string(self, int_seq):
        """Given int seq terminated in STOP, return decoded string."""
        return self.byte_code.to_string(int_seq)

    def __getitem__(self, i):
        return self._to_int_seq(self.string_dataset[i])

    @property
    def num_codes(self):
        """number of byte codes including start and stop."""
        return self.byte_code.num_codes

class ByteDataset(Dataset):

    def __init__(self, fname, byte_code, device=None):
        """fname = path to json db
            byte_code: instance of ByteCode
            device: if not None, where to place tensors
            """
        super().__init__()
        self.fname = fname
        self.byte_code = byte_code
        self.device = device
        self._strds = StringDataset(fname)
        self._bytecodes = ByteWrapper(self._strds, self.byte_code)
    
    def __getitem__(self, i):
        """Returns (seq_len, num_code) one hot float tensor and (seq_ln) int tensor."""
        bts = self._bytecodes[i]
        
        t_onehot = torch.zeros(len(bts), self._bytecodes.num_codes, dtype=torch.float)
        t_onehot[range(len(bts)), bts] = 1
        
        t = torch.tensor(self._bytecodes[i], dtype=torch.long)
        if self.device is not None:
            t_onehot = t_onehot.to(device=self.device)
            t = t.to(device=self.device)
        return t_onehot, t
    
    def __len__(self):
        return len(self._bytecodes)

    def string(self, bytecodes):
        """Returns string corresponding to a list of bytecodes."""
        return self._bytecodes._to_string(bytecodes)

class ByteDataLoader(DataLoader):
    """Loads packed sequences of integer codes for each batch."""

    def __init__(self, byte_ds, pack_onehot=False, **kwargs):
        """pack_onehot: if True, the (seqln, num_codes) input tensors will be packed together into a torch PackedSequence object.
            if False, they're left in a list.
            """
        self.pack_onehot = pack_onehot

        def collate(item_list):
            """item list: a list of (one_hot, int) seq tuples"""
            onehot = [t[0] for t in item_list]
            if self.pack_onehot:
                onehot = pack_sequence(onehot, enforce_sorted=False)
            padded_ints = pad_sequence([t[1] for t in item_list])
            return onehot, padded_ints

        super().__init__(byte_ds, collate_fn=collate, 
                                **kwargs)
    


if __name__ == "__main__":
    bds = ByteDataset("bios.json")
    bytcodes = bds[0]
