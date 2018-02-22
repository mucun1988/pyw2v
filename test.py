import os
import numpy as np
import pandas as pd
from scipy import stats
from pyw2v_u import *
import pickle as pk

make()

num_iter = 5
d_set = [100, 300, 500, 700, 900]
lam_set = [1e-10, 1e-5, 1e-15, 0]

rslt = []
for d in d_set:
    for lam in lam_set:
        print(f"d = {d}.")
        train_w2v_model(num_iter=num_iter, min_count=200,
                        lam=lam, threads=12, word_dim=d)
        vocab, inv_vocab, word_embedding = load_rslt_from_c_output()

        # analogy
        acc = analogical_reasoning(U=word_embedding,
                                   vocab=vocab, inv_vocab=inv_vocab)
        rslt.append({'d': d, 'lam': lam, 'acc': acc['accuracy']})

        df_rslt = pd.DataFrame(rslt)
        print(df_rslt)
        output_file = 'rslt/test_final_submit.csv'
        df_rslt.to_csv(output_file, index=False)

# 10: update once (does not work)
# 11: update more times (plausible)
# 12: fix lambda + update more times ()
# 13: chnage parameter settings: k = 15, downsample=1e-5  (plausible)
# 14: more iterations 5 --> 20 (finally try more iterations + 15 seems bad)
# 15: update once after one epoch (plausible)
# 16: U, V initialized both as normal same (marginally, lamba = 10^-10)
# 17: what others (wired initializations)
# 18: sqrt initializations
# 19: larger than 200 (not good enough)
# 20: longer step size (how)

# linux: /iter + fixed lambda (not good)
# 30: decreasing lambda (good) + normal method w.o, num_thread
# 40: /num_thread (doing)
# 50: method 4
# 504:

# method 3 not working well
