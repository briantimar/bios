# bios

Code for byte-level learning of twitter bios (full training set is in a JSON on my laptop)

[Link](https://drive.google.com/open?id=112BO0CrReSF2DAI4Tn4w8Dq2-XfsjGph) to the ```state_dict``` of a trained model.

```model.py``` - defines the LayerRNN (which was used for training)

```data.py``` - utilties for building byte datasets

```train.py``` - routines used to train the model 

```profiles.py``` - used to build the training set.

## Usage

```sample.py``` is a script for sampling strings from a trained model file, printing to stdout. See command
line help for details. Requires a local copy of a model file (such as that from the link above) plus a model config file and byte values config file; these are available in this repo for the trained model above as ```model_config.json``` and ```byte_values.txt``` respectively.

Example: ```python sample.py model_file -N 5``` prints 5 samples from the model.

Sampling is quite slow on my laptop - GPU will be used automatically if available.