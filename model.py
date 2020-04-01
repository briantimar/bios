import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from data import ByteDataLoader, ByteDataset

class RNN(nn.Module):
    """recurrent character-level model."""

    def __init__(self, hidden_size=256, input_size=206, num_layers=3, dropout=.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                dropout=dropout)
        self.linear = nn.Linear(hidden_size, input_size)
        

    def forward(self, x):
        #outputs at top of LSTM stack
        y = self.lstm(x)[0]
        if isinstance(y, PackedSequence):
            #y is now (max_seqln, batch_size, hidden_size) tensor
            y, lengths = pad_packed_sequence(y)
            logits = self.linear(y).permute(1, 2, 0)
            return logits, lengths
        else:
            logits = self.linear(y).permute(1, 2, 0)
            return logits


if __name__ == "__main__":
    fname = "bios.json"
    ds = ByteDataset(fname)
    dl = ByteDataLoader(ds, batch_size=2)
    rnn= RNN()
    rnn.train()
    onehot, target= next(iter(dl))
    target = target.permute(1, 0)
    lossfn = nn.CrossEntropyLoss(reduction='none')
    logits, lengths = rnn(onehot)
    loss = lossfn(logits, target)
    for i in range(len(lengths)):
        loss[i, lengths[i]:] = 0
    loss = loss.mean()
    loss.backward()


