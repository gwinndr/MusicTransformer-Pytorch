# Music Transformer
Currently supports Pytorch >= 1.2.0 with Python >= 3.6

## About
This is a reproduction of the MusicTransformer (Huang et al., 2018) for Pytorch. This implementation utilizes the generic Transformer implementation introduced in Pytorch 1.2.0 (https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer).

## TODO
* Add music generation results
* Write own midi pre-processor (sustain petal errors with jason's)
   * Support any midi file beyond Maestro
* Fixed length song generation
* Midi augmentations from paper
* Experiment with tensorboard for result reporting

## How to run
You will firstly need to download the Maestro dataset (we used v2 but v1 should work as well). You can download the dataset [here](https://magenta.tensorflow.org/datasets/maestro) (you only need the midi version if you're tight on space). We use the midi pre-processor provided by jason9693 et al. (https://github.com/jason9693/midi-neural-processor) to convert the midi into discrete ordered message types for training and evaluating.

First run third_party/get_code.sh to download the midi pre-processor from github. If on Windows, look at the code and you'll see what to do (it's very simple :D). After, run preprocess_midi.py with --help for details. The result will be a pre-processed folder with a train, val, and test split as provided by Maestro's recommendation.

To train a model, run train.py. Use --help to see the tweakable parameters. See the results section for details on model performance. After training models, you can evaluate them with evaluate.py and generate a midi piece with generate.py. To graph and compare results visually, use graph_results.py.

## Pytorch Transformer
We used the Transformer class provided since Pytorch 1.2.0 (https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer). The provided Transformer assumes an encoder-decoder architecture. To make it decoder-only like the Music Transformer, you use stacked encoders with a custom dummy decoder. This decoder-only model can be found in model/music_transformer.py.

At the time this reproduction was produced, there was no Relative Position Representation (RPR) (Shaw et al., 2018) support in the Pytorch Transformer code. To account for the lack of RPR support, we modified Pytorch 1.2.0 Transformer code to support it. This is based on the Skew method proposed by Huang et al. which is more memory efficient. You can find the modified code in model/rpr.py. This modified Pytorch code will not be kept up to date and will be removed when Pytorch provides RPR support.

## Results
We trained a base and RPR model with the following parameters (taken from the paper) for 300 epochs:
* **learn_rate**: None
* **ce_smoothing**: None
* **batch_size**: 2
* **max_sequence**: 2048
* **n_layers**: 6
* **num_heads**: 8
* **d_model**: 512
* **dim_feedforward**: 1024
* **dropout**: 0.1

![Loss Results Graph](https://lh3.googleusercontent.com/u6AL9vIXG7gBeKuLlVJGFeex7-q2NYLbMqYVZGFI3qxWlpa6hAXdVlOsD52i4jKjrVcf4YZCGBaMIVIagcu_z-7Sg5YhDcgsqcs-p4aR48C287c1QraG0tRnHnmimLd8jizk9afW8g=w2400 "Loss Results")

![Accuracy Results Graph](https://lh3.googleusercontent.com/HGK_UVwa9sbzwJH_myZ3eguMIp1ggww5iMXzCThwf5g0tYRAkfOLK6uykKSuRexmzJDFaea_XpEKP4156gb9HD1nQ8ihJ4BIVehmihiJNQJuf-Uj7dtU7Dk_QWSyhmd6CrgHDjFX2A=w2400 "Accuracy Results")

Best loss for *base* model: 1.99 on epoch 250  
Best loss for *rpr* model: 1.92 on epoch 216

## Discussion
The results were overall close to the results from the paper. Huang et al. reported a loss of around 1.8 for the base and rpr models on Maestro V1. We use Maestro V2 and perform no midi augmentations as they had discussed in their paper. Furthermore, [there are issues with how sustain is handled](https://github.com/jason9693/midi-neural-processor/pull/2) which can be observed by listening to some pre-processed midi files. More refinement with the addition of those augmentations and fixes may yield the loss results in line with the paper.


