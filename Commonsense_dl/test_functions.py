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

if __name__ == '__main__':
    test_wordvec(["pump", "rose", "skill"])

