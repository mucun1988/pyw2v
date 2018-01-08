import os 
import numpy as np
import pandas as pd
from scipy import stats
from pyw2v_u import *

make()

lam_set = [0] + [1/np.power(10, power) for power in np.arange(10,2,-1)] 
rslt = []

for lam in lam_set:
    print(f"lambda = {lam}.")
    train_w2v_model(num_iter=5, min_count=200, lam=lam, threads=5)
    vocab, inv_vocab, word_embedding = load_rslt_from_c_output()
    rslt.append({'lambda': lam}.update(analogical_reasoning(U=word_embedding, vocab=vocab, inv_vocab=inv_vocab)))
    print(pd.DataFrame(rslt))
    print(f"----------------------------------------------------------------------- \n")