import os
import numpy as np
import pandas as pd
from scipy import stats

# os.chdir('F:/matthew/word2vec/pyw2v')


def make():
    """
    compile the c programs
    """
    os.system('gcc word2vec_mm.c -o word2vec_mm -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result')
    os.system(
        'gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result')


def train_w2v(file_to_train='data/enwik9.txt', file_save_vocab='rslt/vocab.txt', file_save_vector='rslt/vocab.txt',
              word_dim=200, window=5, downsample=1e-3, negative=5, threads=12, num_iter=5, min_count=5, alpha=.025, lam=0):
    # set different parameters
    command = './word2vec_mm -cbow 0 -hs 0 -binary 0' + \
        ' -train ' + file_to_train + \
        ' -save-vocab ' + file_save_vocab + \
        ' -output ' + file_save_vector + \
        ' -size ' + str(word_dim) + \
        ' -window ' + str(window) + \
        ' -sample ' + str(downsample) + \
        ' -negative ' + str(negative) + \
        ' -threads ' + str(threads) + \
        ' -iter ' + str(num_iter) + \
        ' -min-count ' + str(min_count) + \
        ' -alpha ' + str(alpha) + \
        ' -lambda ' + str(lam)
    os.system(command)
    # E.g., os.system('./word2vec_mm -train data/enwik9.txt -save-vocab rslt/vocab.txt -output rslt/vector.txt -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 1 -min-count 5000')


def load_rslt_from_c_output(file='rslt/vector.txt'):
    """
    load result from google word2vec 
    """
    df = pd.read_table(file, skiprows=2, header=None, sep=" ")
    n_vocab, dim = df.shape
    dim -= 2
    vocabulary = dict(zip(df.iloc[:, 0].values, range(n_vocab)))
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    U = df.iloc[:, 1:(1 + dim)].values
    return vocabulary, inv_vocabulary, U


def word_vector(word, vocab, U, normalize=True):
    """
    Get vector representation of a word.
    """
    word = word
    if word in vocab:
        vec = U[int(vocab[word])]
        if normalize:
            vec = vec / np.linalg.norm(vec)  # normalize
    else:
        print(f"No such word in vocabulary.")
        vec = None
    return vec


def nearest_words(word, vocab, inv_vocab, U, top=20, display=False):
    """
    Find the nearest words to the word 
    according to the cosine similarity.
    """

    W = normalize2d_row(U)

    if (type(word) == str):
        vec = word_vector(word, vocab, U, normalize=True)
        if vec is not None:
            pass
        else:
            return None
    else:
        vec = word / np.linalg.norm(word)  # normalize

    cosines = (vec.T).dot(W.T)
    args = np.argsort(cosines)[::-1]

    nws = []
    for i in range(1, top + 1):
        nws.append(inv_vocab[args[i]])
        if (display):
            print(inv_vocab[args[i]], round(cosines[args[i]], 3))

    return nws


def normalize2d_row(X):
    return normalize2d_col(X.T).T


def normalize2d_col(X):
    return X / np.linalg.norm(X, axis=0)


def similarity_test_one(U, vocab, data_name='men3000'):
    """
    return the result on similarity test on the dataset
    """
    G = normalize2d_row(U)  # normalize

    filename = data_name + '.csv'
    dataset = pd.read_csv('./data/ws/' + filename,
                          header=None, delimiter=';').values
    ind1 = []
    ind2 = []
    vec2 = []
    model_dict = vocab

    for i in range(dataset.shape[0]):
        word1 = dataset[i, 0].lower()
        word2 = dataset[i, 1].lower()
        if (word1 in model_dict and word2 in model_dict):
            ind1.append(int(model_dict[word1]))
            ind2.append(int(model_dict[word2]))
            vec2.append(np.float64(dataset[i, 2]))

    ind1 = np.array(ind1)
    ind2 = np.array(ind2)
    vec2 = np.array(vec2)

    cosines = (G[ind1] * G[ind2]).sum(axis=1)
    ratio = ind1.shape[0] / dataset.shape[0]

    # return a dict
    return {'dataset': data_name, 'score': stats.spearmanr(cosines, vec2)[0], 'ratio': ratio}


def similarity_test_all(U, vocab):
    """
    return the result on similarity test on all dataset
    """
    sorted_names = ['mc30', 'rg65', 'verb143', 'wordsim_sim', 'wordsim_rel', 'wordsim353',
                    'mturk287', 'mturk771', 'simlex999', 'rw2034', 'men3000']
    rslt = list()

    for xx in sorted_names:

        rslt.append(similarity_test_one(U=U, vocab=vocab, data_name=xx))

    return rslt


vocab, inv_vocab, word_embedding = load_rslt_from_c_output()
word_vector(word='the', vocab=vocab, U=word_embedding, normalize=True)
nearest_words(word='china', vocab=vocab, inv_vocab=inv_vocab,
              U=word_embedding, top=20, display=False)
similarity_test_one(U=word_embedding, vocab=vocab, data_name='men3000')
similarity_test_all(U=word_embedding, vocab=vocab)
