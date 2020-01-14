import numpy as np
import torch

import const


class DataLoader(object):
    def __init__(self, src_texts, src_turn, tgt_indexs, tgt_texts, eos_indexs, src_context, batch_size):
        self.sents_size = len(src_texts)
        self._step = 0
        self.stop_step = self.sents_size // batch_size
        self.batch_size = batch_size
        self.src_texts = src_texts.tolist()
        self.src_turn = src_turn
        self.tgt_indexs = tgt_indexs
        self.tgt_texts = tgt_texts
        self.eos_indexs = eos_indexs
        self.src_context = src_context

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts):
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array(
                [inst + [const.PAD] * (max_len - len(inst)) for inst in insts])
            inst_position = np.array(
                [[pos_i+1 if w_i != const.PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])

            return inst_data, inst_position, max_len

        def index_pairs(t_indexs, tgt_len):
            indexs = np.array([inst.tolist() + [const.PAD] *
                               (tgt_len - len(inst)) for inst in t_indexs])
            return indexs

        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self.batch_size
        _bsz = self.batch_size
        self._step += 1

        src_tensor, src_postion, src_max_len = pad_to_longest(
            self.src_texts[_start:_start+_bsz])
        tgt_tensor, tgt_postion, tgt_max_len = pad_to_longest(
            self.tgt_texts[_start:_start+_bsz])
        tgt_indexs_tensor = index_pairs(
            self.tgt_indexs[_start:_start+_bsz], tgt_max_len)
        turns_tensor = self.src_turn[_start:_start+_bsz]
        eos_indexs = self.eos_indexs[_start:_start+_bsz]
        src_context = self.src_context[_start:_start+_bsz]

        return (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor, src_max_len, eos_indexs, tgt_max_len, src_context


if __name__ == "__main__":
    import common

    corpus = common.middle_load("data/corpus")
    dl = DataLoader(corpus["train"]["src_texts"],
                    corpus["train"]["src_turn"],
                    corpus["train"]["tgt_indexs"],
                    corpus["train"]["tgt_texts"],
                    corpus["train"]["eos_indexs"],
                    corpus["train"]["src_context"],
                    4)
    (src_tensor, src_postion, turns_tensor), (tgt_tensor,
                                              tgt_postion), tgt_indexs_tensor, src_max_len, eos_indexs, tgt_max_len, src_context = next(dl)

    print(tgt_max_len)
    print(tgt_indexs_tensor.tolist())

    ei = eos_indexs[0]
    tgt_ei = tgt_indexs_tensor[0] == 0
    print(tgt_ei)

    print(src_context)
    idx2word = {v: k for k, v in corpus["word2idx"].items()}

    for src in src_tensor:
        print("".join(idx2word[idx] for idx in src))
    for tgt in tgt_tensor:
        print("".join(idx2word[idx] for idx in tgt))
    for index, tgt in enumerate(tgt_indexs_tensor):
        print("".join(idx2word[src_tensor[index][idx]] for idx in tgt))
