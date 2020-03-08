DATAPATH = "./data"

PAD = 0
UNK = 1
BOS = 2
EOS = 3
UNC = 4

WORD = {
    PAD: '&',
    UNK: '#',
    BOS: '^',
    EOS: '$',
    UNC: '@',
}

INIT_RANGE = 0.02
NORM_INIT_STD = 1e-4

SPLIT = '@@^^'
