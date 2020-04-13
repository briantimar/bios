"""Sample bios from a local trained model."""
import argparse
import os
import sys
import json
import torch
from model import LayerRNN
from data import ByteCode

def sample(model_path, model_config, num_samples, 
            byte_value_path, 
            maxlen=200, temperature=1.0):
    """model_path = path to model state dict file.
        model_config = path to model config JSON
        num_samples = number of samples to draw from the model.
        byte_value_path: path to txt holding list of unicode byte values
        maxlen: int, max number of bytes to draw
        temperature: float > 0, sampling temperature. Lower temperature means more 
        conservative sampling"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.", file=sys.stderr)
        return 1
    if not os.path.exists(model_config):
        print(f"Model config file {model_config} not found.", file=sys.stderr)
        return 2
    if not os.path.exists(byte_value_path):
        print(f"Byte value file {byte_value_path} not found.", file=sys.stderr)
        return 2

    bc = ByteCode(byte_value_path)

    with open(model_config) as f:
        config = json.load(f)

    stop_token = config['stop_token']
    model = LayerRNN(config['input_size'], 
                    hidden_size=config['hidden_size'], 
                    num_layers=config['num_layers'], 
                    dropout=config['dropout'])
    
    if torch.cuda.is_available():
        print("Using gpu")
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}.")
    model.eval()
    for __ in range(num_samples):
        bytestring, probs, entropy = model.sample(stop_token, maxlen=maxlen, temperature=temperature )
        print(bc.to_string(bytestring))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="Path to LayerRNN state_dict")
    parser.add_argument('-N', help="Number of samples to draw", type=int, default=1)
    parser.add_argument('--config', help="Path to model config json", type=str, 
                            default="model_config.json")
    parser.add_argument('--byte-values', help="Path to byte value config file", type=str, 
                            default="byte_values.txt", 
                            dest="byte_value_path")
    parser.add_argument('--temperature', type=float, default=1.0, 
                            help="""Temperature to use during sampling. Must be positive float; lower values
                            mean more conservative samples""")
    parser.add_argument('--maxlen', type=int, default=200, 
                            help="Max byte length to sample.")
    args = parser.parse_args()

    sample(args.model, args.config, args.N, args.byte_value_path, 
            temperature=args.temperature, maxlen=args.maxlen)
    