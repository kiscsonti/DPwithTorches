from preprocess import get_data
import utils
def print_params(text, question, a1, a2):
    print("__TEXT__\n", text)
    print(question)
    print(a1[0], " ---->", a1[1])
    print(a2[0], " ---->", a2[1])
    return None


def test_wordvec(words):
    from utils import make_vector_from_words
    for item in make_vector_from_words(words):
        print(len(item), item)


def is_stimm():
    from preprocess import read_vocab
    from preprocess import vocab
    glove_reduced = "data/glove_100d.txt"
    vocab_file = "data/vocab"
    read_vocab(vocab_file)
    counter = 0
    with open(glove_reduced, "r") as f:
        for line in f.readlines():
            counter += 1

    if len(vocab) == counter:
        print("YESSSSSS")

    print(len(vocab), " <==> ", counter)


def test_text_to_grams():
    txt = "What are you doing my boi!"
    txt = "peti"

    partials = utils.text_to_grams(txt, 5)

    print(partials)


if __name__ == '__main__':
    test_text_to_grams()
    #is_stimm()

