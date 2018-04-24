import numpy as np
import pandas as pd
import csv
import operator
#from preprocess import Corpus
from preprocess import get_data, make_vocab, vocab, load_vocab
# glove_data_file = """/media/kiscsonti/521493CD1493B289/egyetem/kutatas/data/glove.6B/glove.6B.100d.txt"""
# model = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

train = "data/train-data.xml"
dev = "data/dev-data.xml"
vocav_file = "data/vocab"

def create_vocab():
    corpus = get_data(train, dev)
    a = corpus.generate()
    i = 0
    while True:
        try:
            if i % 300 == 0:
                print(i)
            item = next(a)

            make_vocab(item, is_nltk=False)
            i += 1

        except StopIteration:
            break

    sorted_x = sorted(vocab, key=vocab.get, reverse=True)

    with open("vocab.txt", "w") as f:
        is_first = True
        for item in sorted_x:

            if is_first:
                is_first = False
            else:
                f.write("\n")
            print(item, vocab[item])
            f.write(item)

    print(len(vocab))


def reduce_glove_file():
    load_vocab(vocav_file)

    print("Read done.")
    from utils import vec
    for item in vocab.keys():

        line = item
        vector = vec(item)
        print(list(vector.columns.values))
        for index, row in vector.iterrows():
            print("nmb: ", row['0'])
            line += " " + str(row['0'])

        line += '\n'
        print(line)
        return
        with open("r_glove.txt", "w+") as f:
            f.write(line)


def main():
    # create_vocab()
    reduce_glove_file()


if __name__ == '__main__':
    main()


