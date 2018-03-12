import csv
import codecs
import random

from utils import normalizeString

def process(_f):
    csvfile = codecs.open(_f, 'r+', 'utf_8_sig')
    reader = csv.reader(csvfile)

    datas = []
    for line in reader:
        if len(line) != 6: continue
        q, d, label = line[3], line[4], line[5]
        q = " ".join(q.strip().split())
        d = " ".join(d.strip().split())
        label = (label.strip())

        datas.append([normalizeString(q), normalizeString(d), label])

    random.shuffle(datas)

    _train = open("data/train", "w")
    _val = open("data/val", "w")
    _split = len(datas) // 10

    [_train.write("\t".join(d) + "\n") for d in datas[:9*_split]]
    [_val.write("\t".join(d) + "\n") for d in datas[9*_split:]]

    _train.close()
    _val.close()

if __name__ == "__main__":
    process("data/train.csv")