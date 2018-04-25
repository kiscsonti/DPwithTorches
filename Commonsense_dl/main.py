from preprocess import get_data
from feature_extractors import to_wordvec
import numpy as np
import utils
import time

sen_len = 10

def main():
    train = "data/train-data.xml"
    dev = "data/dev-data.xml"

    corpus = get_data(train, dev)
    start = time.time()

    # res = np.array(corpus.get_feature(to_wordvec, sen_len))


    print(corpus.get_max_lens())
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
