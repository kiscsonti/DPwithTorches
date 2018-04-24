from preprocess import get_data
from feature_extractors import to_wordvec
import numpy as np
import utils
import time

sen_len = 10

def main():
    train = "train-data.xml"
    dev = "dev-data.xml"

    corpus = get_data(train, dev)
    #print(corpus)


    #print(corpus.get_len_train())
    #print(corpus.hatar)
    from test_functions import print_params

    #corpus.get_feature(print_params)

    start = time.time()

    res = np.array(corpus.get_feature(to_wordvec, sen_len))

    end = time.time()
    print(end - start)

    #utils.make_vector_from_words(["kilo", "brick", "playful"])

if __name__ == '__main__':
    main()