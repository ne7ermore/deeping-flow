## Introduction
This is the implementation for LSTM CNN Text Classfication. <br>
Perform experiments on the English data from [TREC](http://cogcomp.org/Data/QA/QC/)

## Requirement
* python 3.5
* TensorFlow 1.4
* numpy 1.13.1
* tqdm

## Result

> Acc: 88.0%
<p align="center"><img width="40%" src="result.png" /></p>

## Usage
```
python3 train.py -h
```


```
usage: train.py [-h] [--device DEVICE] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--seed SEED] [--data DATA]
                [--lr LR] [--dropout DROPOUT] [--emb_dim EMB_DIM]
                [--hidden_sizes HIDDEN_SIZES] [--l_2 L_2]
                [--filter_sizes FILTER_SIZES] [--num_filters NUM_FILTERS]

LSTM CNN Classification

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --seed SEED
  --data DATA
  --lr LR
  --dropout DROPOUT
  --emb_dim EMB_DIM
  --hidden_sizes HIDDEN_SIZES
  --l_2 L_2
  --filter_sizes FILTER_SIZES
  --num_filters NUM_FILTERS
```