from preprocess import get_data

def main():
    train = "train-data.xml"
    dev = "dev-data.xml"

    corpus = get_data(train, dev)
    print(corpus)


    print(corpus.get_len())
    print(corpus.hatar)

if __name__ == '__main__':
    main()