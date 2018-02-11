import os
import numpy as np
import pandas as pd
from scipy import stats
from pyw2v_u import *
import pickle as pk

make()

lam_set = [0] + [1 / np.power(10, power) for power in np.arange(15, 2, -1)]
# lam_set = [.00001]
rslt = []
d = 200

for lam in lam_set:
    print(f"lambda = {lam}.")
    train_w2v_model(num_iter=15, min_count=5,
                    lam=lam, threads=5, word_dim=d)
    vocab, inv_vocab, word_embedding = load_rslt_from_c_output()
    # analogy
    xx = {'lambda': lam}
    xx.update(analogical_reasoning(U=word_embedding,
                                   vocab=vocab, inv_vocab=inv_vocab))
    # similarity
    rslt.append(xx)

    for zz in similarity_test_all(U=word_embedding, vocab=vocab):
        yy = {'lambda': lam}
        yy.update(zz)
        rslt.append(yy)

    print(pd.DataFrame(rslt))
    print(f"----------------------------------------------------------------------- \n")

    df_rslt = pd.DataFrame(rslt)
    output_file = 'rslt/result_v6_' + str(d) + '.pkl'
    df_rslt.to_pickle(output_file)
