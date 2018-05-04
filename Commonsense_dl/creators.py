from preprocess import get_data


def create_char_indexes():

    train = "data/train-data.xml"
    dev = "data/dev-data.xml"

    corpus = get_data(train, dev)

    char_set = set()
    generator = corpus.generate_train()

    for item in generator:
        # item[0] -> text, item[1] -> question, item[2]-> asnwer 1, Item[3] -> answer 2
        for char in item[0]:
            if char not in char_set:
                char_set.add(char)

        for char in item[1]:
            if char not in char_set:
                char_set.add(char)

        for char in item[2]:
            if char not in char_set:
                char_set.add(char)

        for char in item[3]:
            if char not in char_set:
                char_set.add(char)

    print(char_set)

    with open("data/char_vocab.txt", "w") as f:
        first = True
        for item in char_set:
            if first:
                first = False
            else:
                f.write("\n")
            f.write(item)


if __name__ == '__main__':
    create_char_indexes()
