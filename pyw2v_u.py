import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import re

# os.chdir('F:/matthew/word2vec/pyw2v')


def download_preprocess_data(output_file='data/enwik9', replace=True):
    if Path(output_file).is_file():
        if not replace:
            print(f'file exists and no replacement... done.')
            return None
        else:
            print(f'file exists but now replace it.')
            os.system('rm ' + output_file)
    else:
        print(f'file does not exist and now create it.')
    print(f'download data...')
    # os.system('mkdir data')
    os.system('mkdir rslt')  # store output
    os.system('wget http://mattmahoney.net/dc/enwik9.zip -O data/tmp.gz')
    print(f'download data... done')
    print(f'preprocess the data and save it at {output_file}...')
    os.system('gzip -d data/tmp.gz -f')
    os.system('perl wikifil.pl data/tmp > data/text')
    with open("data/text", 'r') as myfile:
        data = myfile.read()
    sentences = data.split('.')
    f = open(output_file, 'w')
    for sentence in sentences:
        if len(sentence.split()) > 0:
            f.write("%s \n" % sentence)
    f.close()
    print(f'preprocess the data and save it at {output_file}... done.')
    os.system('rm data/tmp')
    os.system('rm data/text')


def make():
    """
    compile the c programs
    """
    os.system('gcc word2vec_mm.c -o word2vec_mm -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result')
    os.system(
        'gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result')


def train_w2v_model(file_to_train='data/enwik9', file_save_vocab='rslt/vocab.txt', file_save_vector='rslt/vector.txt',
                    word_dim=100, window=5, downsample=1e-3, negative=5, threads=12, num_iter=5, min_count=5, alpha=.025, lam=0):
    """
    train google word2vec model (default settings are provided)
    """
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
    # words = [str(xx) for xx in df.iloc[:,0].values]
    # vocabulary = dict(zip(words, range(n_vocab)))
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    U = df.iloc[:, 1:(1 + dim)].values
    return vocabulary, inv_vocabulary, U


def word_vector(word, vocab, U, normalize=True):
    """
    get vector representation of a word.
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


def nearest_words(word, vocab, inv_vocab, U, top=20):
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
    else:  # the input could be an array
        vec = word / np.linalg.norm(word)  # normalize
    cosines = (vec.T).dot(W.T)
    args = np.argsort(cosines)[::-1]
    nws = []
    for i in range(1, top + 1):
        nws.append({'word': inv_vocab[args[i]],
                    'score': round(cosines[args[i]], 3)})
    return nws


def normalize2d_row(X):
    """
    normalize the rows of X
    """
    return normalize2d_col(X.T).T


def normalize2d_col(X):
    """
    normalize the columns of X
    """
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
    return {'dataset': data_name, 'accuracy': stats.spearmanr(cosines, vec2)[0], 'ratio': ratio}


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


def find_b_star(tri_words, U, vocab, inv_vocab, normalized=True):
    """
    to predict b*
    tri_words: (a, a*, b)
    """
    if normalized:
        W = U * 1.0
    else:
        W = normalize2d_row(U)  # do the normalization

    if (tri_words[0] in vocab) and \
       (tri_words[1] in vocab) and \
       (tri_words[2] in vocab):

        indices = [vocab[x] for x in tri_words]
        words3_vec = W[indices]
        cosines = W.dot(words3_vec.T) / 2 + .5
        obj = (cosines[:, 1] * cosines[:, 2]) / (cosines[:, 0] + 1e-3)
        pred_idx = obj.argsort()[-4:][::-1]
        for idx in pred_idx:
            if idx not in indices:
                return inv_vocab[idx]
    else:
        #print('At least one of the three words is not in the vocab.')
        return None


def analogical_reasoning(U, vocab, inv_vocab):

    W = normalize2d_row(U)  # do the normalization

    google = './data/analogy/google.txt'
    dataset = pd.read_csv(google, header=None, delimiter='\t').values

    good_sum = 0
    miss_sum = 0

    for words in dataset:
        a, a_, b, b_ = words
        tri_words = (a, a_, b)

        if (a in vocab) and (a_ in vocab) and (b in vocab) and (b_ in vocab):
            pred = find_b_star(tri_words, W, vocab, inv_vocab,
                               normalized=True)  # prediction
            if (pred == b_):
                good_sum += 1
        else:
            miss_sum += 1

    # calculate accuracy
    acc = good_sum / float(dataset.shape[0] - miss_sum)
    ratio = 1 - float(miss_sum) / dataset.shape[0]

    return {'dataset': 'google', 'accuracy': acc, 'ratio': ratio}


if __name__ == "__main__":
    vocab, inv_vocab, word_embedding = load_rslt_from_c_output()
    word_vector(word='the', vocab=vocab, U=word_embedding, normalize=True)
    xx = nearest_words(word='china', vocab=vocab,
                       inv_vocab=inv_vocab, U=word_embedding, top=20)
    similarity_test_one(U=word_embedding, vocab=vocab, data_name='men3000')
    similarity_test_all(U=word_embedding, vocab=vocab)

    tri_words = ('boy', 'girl', 'king')
    find_b_star(tri_words=tri_words, U=word_embedding,
                vocab=vocab, inv_vocab=inv_vocab, normalized=False)

    analogical_reasoning(U=word_embedding, vocab=vocab, inv_vocab=inv_vocab)
