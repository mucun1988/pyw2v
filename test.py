import os
import numpy as np
import pandas as pd
from scipy import stats
from pyw2v_u import *
import pickle as pk

make()  # compile the C code

dim = 300
lam = 1e-10

printf('training sgns-qr model...')
train_w2v_model(min_count=200, lam=lam, word_dim=dim, threads=12)
printf('training sgns-qr model... done. \n')

printf('load the result and evaluate the performance...')
vocab, inv_vocab, word_embedding = load_rslt_from_c_output()
acc = analogical_reasoning(U=word_embedding,
                           vocab=vocab, inv_vocab=inv_vocab)
printf('load the result and evaluate the performance... \n')

rslt = {'dim': dim, 'lam': lam, 'acc': acc['accuracy']}

print(rslt)
