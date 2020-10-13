# Neuronal Gaussian Process Regression

This repository is the official implementation of [Neuronal Gaussian Process Regression](https://papers.nips.cc/paper/9932-neuronal-gaussian-process-regression). 

![Image of NN](https://github.com/j-friedrich/neuronalGPR/blob/master/fig.png)

## Requirements

To install requirements (using [conda](https://www.anaconda.com/products/individual)) and download the datasets, execute:

```setup
conda env create -f environment.yml
conda activate neuronalGPR
python setup.py
```

## Training

Pre-trained models are included in the results directory of this repository.
To nevertheless re-train the models on the UCI datasets, run these commands:

```train
python runUCI.py <UCI Dataset directory>
python PBP.py <UCI Dataset directory> <number of hidden layers>
python Dropout.py <UCI Dataset directory> <number of hidden layers>
```

## Evaluation

To reproduce the figures, run

```fig
python <name_of_fig_script.py>
```

To reproduce the tables, run

```table
python table1.py; python table2.py; 
```

## Pre-trained Models

The pre-trained models are included in the results directory of this repository.
