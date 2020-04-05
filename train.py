import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from data import ByteDataLoader, ByteDataset, ByteCode
from torch.optim import Adam
from model import RNN
from datetime import datetime
import os
import time
import json

def train(dataloader, model, optimizer, params, device, 
            byte_code):
    """Train with the given model and optimizer under the setting prescribed in params.
        dataloader = ByteDataLoader instance
        model = RNN
        optimizer = instance of torch.optim with model params loaded.
        params: dict holding hyperparams etc for training.
        device = a gpu, hopefully
        byte_code: instance of ByteCode, used for producing samples
        output_dir = where to write outputs of training.
        """
    epochs = params["epochs"]
    expt_dir = params["expt_dir"]
    sample_step = params.get("sample_step", 100)
    tstart = datetime.now()
    byte_samples = []
    string_samples = []
    probabilities = []
    entropies = []
    losses = []

    lossfn = nn.CrossEntropyLoss(reduction='none')

    try:
        for ep in range(epochs):
            for batch_index, (onehot, target) in enumerate(dataloader):

                model.train()
                logits, lengths = model(onehot)

                loss = lossfn(logits[...,:-1], target[...,1:])
                for i in range(len(lengths)):
                    loss[i, lengths[i]-1:] = 0
                    
                loss = loss.sum(dim=1).mean()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                _loss = loss.detach().cpu().item()
                
                losses.append(_loss)
                if batch_index % sample_step == 0:
                    model.eval()
                    bytestring, probs, entropy = model.sample(byte_code.STOP_CODE,maxlen=200, temperature=1.0)
                    str_sample = byte_code.to_string(bytestring)

                    byte_samples.append(bytestring)
                    string_samples.append(str_sample)
                    probabilities.append(probs)
                    entropies.append(entropy)
                    print(f"Step {batch_index}, sample: {str_sample}")
                    print(f"recent loss: {loss:.3f}")

            # save each epoch
            model_fname = os.path.join(expt_dir, f"model_epoch_{ep}")
            opt_fname = os.path.join(expt_dir, f"opt_epoch_{ep}")
            torch.save(model.state_dict(), model_fname)
            torch.save(optimizer.state_dict(), opt_fname)
    finally:
        tend = datetime.now()
        expt_data = {'loss': losses, 'byte_samples': byte_samples, 'string_samples': string_samples, 
                    'entropies': entropies, 
                    'tstart': str(tstart), 'tend': str(tend),
                    **params}
        with open(os.path.join(expt_dir, "expt_data"), 'w') as f:
            json.dump(expt_data, f)
        
    

if __name__ == "__main__":

    fname = "_bios.json"
    bc = ByteCode("byte_values.txt")
    ds = ByteDataset(fname, bc, device=torch.device('cpu'))
    print(f"Loaded {len(ds)} samples")
    dl = ByteDataLoader(ds, batch_size=1)
    rnn= RNN(bc.num_codes)
    rnn.train()
    epochs = 1
    lr=1e-3
    losses = []
    lossfn = nn.CrossEntropyLoss(reduction='none')

    optimizer = Adam(rnn.parameters(), lr=lr)

    train(dl, rnn, optimizer, dict(epochs=epochs, expt_dir="tst",sample_step=1), torch.device('cpu'), 
            bc)