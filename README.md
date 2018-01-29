# Differential and Cooperative Binding Feature Discovery with Convolutional Neural Networks

This repository contains pre-processed datasets and code required to replicate the results from the paper by Phuycharoen et al.

You can install the python dependencies with pip. First install:

```
sudo pip install numpy scipy sklearn twobitreader matplotlib h5py
```

then the remaining packages:

```
sudo pip install GPy tensorflow keras 
```

To use GPU acceleration make sure you installed CUDA drivers and tensorflow-gpu beforehand. Tensorflow backend for Keras is required.

## Usage:

Run Main.py to recreate the plots visualised in the paper using pre-trained models.

Removing (or re-naming) the 'models' folder will train the models from scratch using the hyper-parameters contained in the 'hyper' folder. If the 'hyper' folder is also removed, a hyper-parameter search will be performed. This may take many days depending on your system configuration.
