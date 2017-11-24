## SeqMatchSeq
Module implemention from "[A Compare-Aggregate Model for Matching Text Sequences](https://arxiv.org/abs/1611.01747)"

## Requirement
* python 3.5
* TensorFlow 1.4
* numpy 1.13.1
* tqdm

## Usage
```
python3 train.py -h
```

You will get:

```
usage: train.py [-h] [--device DEVICE] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--seed SEED] [--save SAVE]
                [--data DATA] [--not-use-w2v] [--w2v-file W2V_FILE] [--lr LR]
                [--dropout DROPOUT] [--attn-dim ATTN_DIM] [--emb-dim EMB_DIM]
                [--l_2 L_2] [--eps EPS] [--filter-sizes FILTER_SIZES]
                [--num-filters NUM_FILTERS]

SeqMatchSeq

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       train device
  --epochs EPOCHS       number of epochs for train
  --batch-size BATCH_SIZE
                        batch size for training
  --seed SEED           random seed
  --save SAVE           path to save the final model
  --data DATA           location of the data corpus
  --not-use-w2v         no word2vec
  --w2v-file W2V_FILE   pre trained word2vec
  --lr LR               initial learning rate
  --dropout DROPOUT     the probability for dropout (0 = no dropout)
  --attn-dim ATTN_DIM   preprocess dim
  --emb-dim EMB_DIM     number of embedding dimension
  --l_2 L_2             l_2 regularization
  --eps EPS
  --filter-sizes FILTER_SIZES
                        filter sizes
  --num-filters NUM_FILTERS
                        number of filters
```

## Train
```
python3 train.py
```

