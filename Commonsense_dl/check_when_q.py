from preprocess import get_data, get_tokenized_text
def main():
    q_word = "when"
    train = "data/train-data.xml"
    dev = "data/dev-data.xml"

    corpus = get_data(train, dev)

    generator = corpus.generate_dev()

    get_match(generator)


    pass

def get_match(generator):
    boopity_boop = {"in", "on", "the", "from", "an", "a", "at", ".", "?", "'s", "their", } #kivételek amiket nem kell számolni
    counter = 0
    counf_of_alls = 0
    sum_of_alls = 0
    for g in generator:
        tokenized_q = list(map(str.lower, get_tokenized_text(g[1])[0].split()))

        if tokenized_q.__contains__("where"):
            print(tokenized_q)
            tokenized_t = list(map(str.lower, get_tokenized_text(g[0])[0].split()))
            tokenized_a1 = list(map(str.lower, get_tokenized_text(g[2])[0].split()))
            tokenized_a2 = list(map(str.lower, get_tokenized_text(g[3])[0].split()))

            score_1, all_1 = similarity(tokenized_t, tokenized_a1)
            score_2, all_2 = similarity(tokenized_t, tokenized_a2)

            if score_1/all_1 == 1:
                counf_of_alls += 1

            sum_of_alls += score_1/all_1

            if score_2/all_2 == 1:
                counf_of_alls += 1

            sum_of_alls += score_2/all_2
            counter += 2
            print("Answer 1: ", tokenized_a1)
            print("Answer 2: ", tokenized_a2)
            if counter % 50 == 0:
                print("Count of all Ones: ", (counf_of_alls / counter) * 100, "%")
                print("Sum of all: ", (sum_of_alls / counter))
                print("counter: ", counter)

    print("Count of all Ones: ", (counf_of_alls/counter)*100, "%")
    print("Sum of all: ", (sum_of_alls/counter))
    print("counter: ", counter)


def similarity(text, answer):
    found = 0
    for word in answer:
        if word in text:
            found += 1
    return found, len(answer)

if __name__ == '__main__':
    main()
