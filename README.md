# Music Transformer
Currently supports Pytorch 1.2.0 with Python >= 3.6

Disclaimer: Still a work in progress :)

## About
This is a reproduction of the MusicTransformer (Huang et al., 2018) for Pytorch. This implementation utilizes the generic Transformer implementation introduced in Pytorch 1.2.0 (https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer). Statistics gathering is still in progress and this README will be updated upon completion.

## How to run
You will firstly need to download the Maestro dataset (we used v2 but v1 should work as well). You can download the dataset [here](https://magenta.tensorflow.org/datasets/maestro) (you only need the midi version if you're tight on space). We use the midi pre-processor provided by jason9693 et al. (https://github.com/jason9693/midi-neural-processor) to convert the midi into discrete ordered message types for training and evaluating.

First run third_party/get_code.sh to download the midi pre-processor from github. If on Windows, look at the code and you'll see what to do (it's very simple :D). After, run preprocess_midi.py with --help for details. The result will be a pre-processed folder with a train, val, and test split as provided by Maestro's recommendation.

To train a model, run train.py. Use --help to see the tweakable parameters. See the results section (TODO) for the results on various configurations we tested. After training models, you can evaluate them with evaluate.py and generate a midi piece with generate.py.

## Pytorch Transformer
We used the Transformer class provided since Pytorch 1.2.0 (https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer). The provided Transformer assumes an encoder-decoder architecture. To make it decoder-only like the Music Transformer, you use stacked encoders with a custom dummy decoder. This decoder-only model can be found in model/music_transformer.py.

At the time this reproduction was produced, there was no Relative Position Representation (RPR) (Shaw et al., 2018) support in the Pytorch Transformer code. To account for the lack of RPR support, we modified Pytorch 1.2.0 Transformer code to support it. This is based on the Skew method proposed by Huang et al. which is more memory efficient. You can find the modified code in model/rpr.py. This modified Pytorch code will not be kept up to date and will be removed when Pytorch provides RPR support.
