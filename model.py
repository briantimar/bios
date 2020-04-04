import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from data import ByteDataLoader, ByteDataset
from torch.distributions import Categorical

class RNN(nn.Module):
    """recurrent character-level model."""

    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                dropout=dropout)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """x : a tensor of integer bytecodes."""
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

    def _get_weights(self, i):
        """Get weights for the cell of layer i.
            Does not copy!"""
        return {'weight_ih': getattr(self.lstm, f"weight_ih_l{i}"), 
                'weight_hh': getattr(self.lstm, f"weight_hh_l{i}"), 
                'bias_ih': getattr(self.lstm, f"bias_ih_l{i}"), 
                'bias_hh': getattr(self.lstm, f"bias_hh_l{i}")}

    def _get_cell(self, i, device=None):
        """ Returns LSTM cell module loaded with weights from the ith LSTM layer.
            Device is same as the model."""
        if not (0<= i < self.num_layers):
            raise ValueError(f"invalid layer index {i}")
        if i == 0:
            cell = nn.LSTMCell(self.input_size, self.hidden_size)
        else:
            cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        wts = self._get_weights(i)
        for wt in wts:
            getattr(cell, wt).data.copy_(wts[wt].data)
        if device is None:
            device = wts[wt].device
        return cell.to(device=device)

    def get_cells(self, device=None):
        """ Copies lstm params into a list of LSTM cells.
            returns: list whose ith element is the cell of the ith layer."""
        return [self._get_cell(i, device=device) for i in range(self.num_layers)]
    
    @property
    def device(self):
        """current weight device."""
        return self.linear.weight.device

    def sample(self, stop_token, maxlen=400, temperature=1.0):
        """Sample a string of bytecodes from the model distribution. Sampling halts 
        start_token, int: token with which to start the string; fed into leftmost cell
        stop_token, int: token upon receipt of which sampling halts.
        maxlen = max allowed sample length.
        temperature: scaling factor by which the logits are divided prior to sampling. A low 
        temperature makes samping converge on most likely characters.
        returns: a list of integer bytecodes
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not (0<= stop_token < self.input_size):
            raise ValueError(f"Not a valid stop token: {stop_token}")

        cells = self.get_cells()
        
        output = stop_token
        bytestring = [output]
        #record the uncertainty in the model output at each step.
        entropies = []
        #and the probability of the selected output
        probs_sampled = []
        #track the hidden and cell states at each sequence step
        # they default to zero in the pytorch impl, so can start as None
        h = [None] * self.num_layers
        c = [None] * self.num_layers
        while len(bytestring) < maxlen - 1 and (len(bytestring)==1 or (output != stop_token)):
            #feed previous output as input
            inp = torch.zeros((1,self.input_size,), dtype=torch.float)
            inp[0,output] = 1
            inp=inp.to(device=self.device)
            
            for i in range(self.num_layers):
                # at the first timestep
                if h[i] is None:
                    h[i], c[i] = cells[i](inp)
                # at all subsequent
                else:
                    h[i], c[i] = cells[i](inp, (h[i], c[i]))
                inp = h[i]

            # apply linear to the upper hidden state and sample from the byte distribution.
            
            logits = self.linear(h[self.num_layers-1]).squeeze() / temperature
            probs = logits.softmax(0).detach()
            probs[probs<1e-12] = 1e-12
            entropies.append(-(probs * probs.log2()).sum().item())
            output = Categorical(logits=logits).sample().item()
            bytestring.append(output)
            probs_sampled.append(probs[output].item())

        if len(bytestring) == maxlen - 1:
            print(f"Warning - max length {maxlen} reached")
            if bytestring[-1] != stop_token:
                bytestring.append(stop_token)

        return bytestring, probs_sampled, entropies


if __name__ == "__main__":
    from data import ByteCode
    byte_code = ByteCode("byte_values.txt")
    model = RNN(input_size=byte_code.num_codes)
    b,p,e = model.sample(byte_code.STOP_CODE, maxlen=20)
    print(byte_code.to_string(b))

  


