# PL-CNN
An optimization algorithm for learning Piecewise Linear Convolutional Neural Networks.
This is the implementation of the paper [Trusting SVM for Piecewise Linear CNNs](https://arxiv.org/abs/1611.02185) by Leonard Berrada, Andrew Zisserman and M. Pawan Kumar.

# Requirements
This code has been developped python 2.7 and requires the packages given in `requirements.txt`. Note that you might need to install the dev version of Lasagne and Theano.

# Repository organization

## data

Contains the data set folders. The use of the following data sets is implemented:
* MNIST
* CIFAR-10
* CIFAR-100
* ImageNet

## src

This contains the python code. To run the tests, run `sh scripts/test.sh` from the `src` working directory.

## log

This contains the logs from the experiments as well as the saved models. All experiment logs are backed up in `log/all`.

# To reproduce results

First make sure your working directory is `src`, and that the standard data sets are in `data`.
Then to run experiments with gpu [x] with results output to directory `log/my_dir`:
## MNIST:
`sh scripts/mnist.sh gpu[x] my_dir`

## CIFAR-10:
`sh scripts/cifar10.sh gpu[x] my_dir`

## CIFAR-100:
`sh scripts/cifar100.sh gpu[x] my_dir`

## ImageNet:
The following code assumes that:
* the images have been extracted and resized to a minimal size of 256, and are in the path specified in `datasets/imagenet.py`.
* the pre-trained VGG-16 model has been downloaded from [Lasagne model zoo](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl) and placed in `my_dir`, the directory for the experiment.

Launch data pipeline for training set:
`python server.py --train`

And in a different shell:
`python lwsvm.py --dataset imagenet --in_name vgg16 --device gpu[x] --xp_dir my_dir`

When training is complete, launch data pipeline for validation set:
`python server.py --val`

And launch evaluation in a different shell:
`python extract.py --imagenet --weights_file imagenet_lwsvm_15`

# To re-use code

The architectures can easily be changed through the `addLayer`-type methods of `NeuralNet`. Note that the code assumes a restricted search on the conditional gradient of convolutional layers (Observation 1 in section 4 of the [paper](https://arxiv.org/abs/1611.02185)), since they are usually followed by an important number of non-linearities (max-pooling).



