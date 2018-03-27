import numpy as np
from const import PAD

def _lcs(x, y):
    n = len(x)
    m = len(y)
    table = dict()

    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
            else:
                table[i, j] = max(table[i-1, j], table[i, j-1])

    def recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i-1] == y[j-1]:
            return recon(i-1, j-1) + [x[i-1]]
        elif table[i-1, j] > table[i, j-1]:
            return recon(i-1, j)
        else:
            return recon(i, j-1)

    return len(recon(n, m)), n, m

def rouge_l(evals, refs):
    assert evals.shape == refs.shape

    scores = []
    for eva, ref in zip(evals, refs):
        same_len, eva_len, ref_len = map(float, _lcs(eva, ref[np.where(ref>PAD)]))

        r_lcs, p_lcs = same_len/ref_len, same_len/eva_len

        beta = p_lcs / (r_lcs + 1e-12)
        f_lcs = ((1 + (beta**2)) * r_lcs * p_lcs) / (r_lcs + ((beta**2) * p_lcs) + 1e-12)
        scores.append(f_lcs)

    scores = np.asarray(scores, dtype=np.float32)
    scores = np.repeat(scores[:, np.newaxis], evals.shape[1], 1)

    return scores


if __name__ == '__main__':
    data = np.asarray([[0,0,0,0,0,0],[4,4,4,4,0,0]], dtype=np.int64)
    label = np.asarray([[3,1,2,3,1,0],[2,3,2,3,1,0]], dtype=np.int64)

    print(rouge_l(data, label))
