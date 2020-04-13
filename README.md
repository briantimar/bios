# bios

Code for byte-level learning of twitter bios (full training set is in a JSON on my laptop)

[Link](https://drive.google.com/open?id=112BO0CrReSF2DAI4Tn4w8Dq2-XfsjGph) to the ```state_dict``` of a trained model.

```model.py``` - defines the LayerRNN (which was used for training)

```data.py``` - utilties for building byte datasets

```train.py``` - routines used to train the model 

```profiles.py``` - used to build the training set.