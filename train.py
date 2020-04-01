import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from data import ByteDataLoader, ByteDataset
from torch.optim import Adam
from model import RNN
import time

if __name__ == "__main__":

    fname = "_bios.json"
    ds = ByteDataset(fname)
    print(f"Loaded {len(ds)} samples")
    dl = ByteDataLoader(ds, batch_size=64)
    rnn= RNN()
    rnn.train()
    epochs = 5
    lr=1e-3
    losses = []
    lossfn = nn.CrossEntropyLoss(reduction='none')

    optimizer = Adam(rnn.parameters(), lr=lr)

    for ep in range(epochs):
        for onehot, target in dl:
            t0 = time.time()
            target = target.permute(1, 0)
            logits, lengths = rnn(onehot)
            loss = lossfn(logits, target)
            for i in range(len(lengths)):
                loss[i, lengths[i]:] = 0
            loss = loss.mean()
            rnn.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            _loss = loss.detach().item()
            print(f"loss {_loss} in {t1 - t0:.3f} sec")
            losses.append(loss)
    
