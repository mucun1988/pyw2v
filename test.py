import os
import numpy as np
import pandas as pd
from scipy import stats
from pyw2v_u import *
import pickle as pk

make()

num_iter = 5
d_set = [100, 200, 300, 400, 500]
lam_set = [0, 1e-5, 1e-10, 1e-15]
rslt = []
for d in d_set:
    for lam in lam_set:
        print(f"d = {d}.")
        train_w2v_model(num_iter=num_iter, min_count=200,
                        lam=lam, threads=8, word_dim=d)
        vocab, inv_vocab, word_embedding = load_rslt_from_c_output()

        # analogy
        acc = analogical_reasoning(U=word_embedding,
                                   vocab=vocab, inv_vocab=inv_vocab)
        rslt.append({'d': d, 'lam': lam, 'acc': acc['accuracy']})

        df_rslt = pd.DataFrame(rslt)
        print(df_rslt)
        output_file = 'rslt/test_v10_iter_' + str(num_iter) + '.csv'
        df_rslt.to_csv(output_file, index=False)

# 10: update once (possible)
# 11: update more times (possible)
# 12: fix lambda + update more times (sees that fix lambda not working)
# 13: chnage parameter settings: k = 15, downsample=1e-5  (seems not working, more iterations?)
# 14: more iterations 5 --> 20 (finally try more iterations)
