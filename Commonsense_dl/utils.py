import spacy
import numpy as np
import pandas as pd
import csv
import preprocess
import json
import doc
from preprocess import char_vocab
import config

# model = spacy.load('en_core_web_sm')
glove_data_file = "data/glove.6B.100d.txt"
# glove_data_file = """/media/kiscsonti/521493CD1493B289/egyetem/kutatas/data/glove.6B/glove.6B.100d.txt"""
model = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
processed_train = "data/my_processed_train.json"
processed_dev = "data/my_processed_dev.json"


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


def load_data(filename):
    records = []

    with open(filename, "r") as f:
        for line in f.readlines():
            records.append(doc.Example(json.loads(line)))
    return records


def text_to_grams(text):
    length = config.args.n_grams
    partials = []

    if len(text) < length:
        partials.append(text)
    else:
        for i in range(length, len(text)):
            partials.append(text[i-length:i])

    return partials


def text_to_char_index(text):
    indexed = []
    for char in text:
        indexed.append(char_vocab[char])
    return indexed


if __name__ == '__main__':
    text_to_char_index("petigep hahah")
