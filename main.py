import os 
import numpy as np
import pandas as pd
from scipy import stats
from pyw2v_u import *

make()

train_w2v_model(num_iter=20, min_count=5)