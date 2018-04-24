import spacy
import numpy as np
import pandas as pd
import csv


#model = spacy.load('en_core_web_sm')
glove_data_file = """/media/kiscsonti/521493CD1493B289/egyetem/kutatas/data/glove.6B/glove.6B.100d.txt"""
model = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


def make_vector_from_words(words):
    vecs = []
    for word in words:
        w = vec(word.lower())
        vecs.append(w)
    return vecs


def vec(w):
    try:
        v = model.loc[w].as_matrix()
    except (KeyError):
        v = np.zeros(100)
    return v


def extend_array(target_size, arr):
    if len(arr) <= target_size:
        while len(arr) < target_size:
            arr.append(np.zeros(np.array(arr[0]).shape).tolist())
    else:
        return arr[:target_size]
    return arr


