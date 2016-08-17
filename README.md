# deep-features
Extraction of features from a convolutional neural network (VGG, AlexNet, etc.) using torch

## Installation
    git clone https://github.com/kidzik/deep-features.git

## Preparation
Download VGG (or other model)

    cd deep-features/models
    ./download_models.sh

## Execution
    th extract.lua -images_path PATH_TO_IMAGES

The script will go through all images, get the output of the neural network and save it to a CSV file output.csv.

## Parameters
* `-images_path`: Path to images
* `-output_csv`: Path to output CSV
* `-proto_file`:, Path to prototxt file of the model
* `-model_file`: Path to caffemodel
* `-ext`: Extension of images
