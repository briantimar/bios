import torch
import unittest
from model import RNN


class TestRNN(unittest.TestCase):

    def test__rnn_forward(self):
        input_size =5
        hidden_size = 32
        rnn = RNN(input_size, hidden_size=hidden_size)
        lengths = torch.tensor([4, 3, 3, 2])
        batch_size = len(lengths)
        inputs = [torch.randn(lengths[i], input_size) for i in range(batch_size)]

        outputs, lens = rnn._rnn_forward(inputs, truncation_length=None)
        self.assertEqual(outputs.shape, (4, batch_size, hidden_size ))

        #check that gradients are flowing
        outputs.mean().backward()
        for i in range(rnn.num_layers):
            self.assertTrue(rnn.cells[i].weight_hh.grad.abs().sum() > 0)

        full_outputs = outputs.detach().clone()
        # now check that gradients can be truncated without affecting the output
        # of the forward pass
        trunc_outputs, lens = rnn._rnn_forward(inputs, truncation_length=2)
        self.assertEqual(trunc_outputs.shape, full_outputs.shape)
        self.assertAlmostEqual((trunc_outputs.data - full_outputs.data).abs().sum().item(), 0)

    def test_forward(self):
        input_size =5
        hidden_size = 32
        rnn = RNN(input_size, hidden_size=hidden_size)
        lengths = [5, 3,4, 2,4]
        batch_size = len(lengths)
        inputs = [torch.randn(lengths[i], input_size) for i in range(batch_size)]
        logits, lens = rnn.forward(inputs)
        self.assertEqual(lens, sorted(lengths, key = lambda x: -x))
        self.assertEqual(logits.shape, (batch_size, input_size, max(lengths)))

    def test_sample(self):
        input_size =5
        hidden_size = 32
        rnn = RNN(input_size, hidden_size=hidden_size)
        stop_token = 4
        bytestring, probs, entropies = rnn.sample(stop_token, maxlen=20)
        self.assertTrue(max(bytestring) < rnn.input_size)
        self.assertTrue(min(bytestring) >= 0)


if __name__ == "__main__":
    unittest.main()
