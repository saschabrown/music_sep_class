# Using Machine Learning to classify and separate musical instruments
This project explores the use of machine learning algorithms to distinguish different musical instruments. The project uses data from the NSynth dataset, which can be found at:

    https://magenta.tensorflow.org/datasets/nsynth

The goals of the project are divided in 3 steps:

    1. Classify the type of instrument from a .wav file with a single instrument
    2. Classify the types of intsruments from a .wav file with multiple instruments
    3. Separate a .wav file with multiple instruments into various .wav files of a single instrument

## Setup of the directory
The directory contains 3 main folders:

    1. The functions folder contains the definition of functions used in the other scripts
    2. The training_scripts folder contains the scripts that were used to train the models, subdivided in the 3 goals
    3. The models folder contains the weights of the trained models, ready to be applied to testing data

## Idea, code and paper for Multi Channel U-net for music seperation
https://vskadandale.github.io/multi-channel-unet/
