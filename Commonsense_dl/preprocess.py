from xml.etree import ElementTree
import unicodedata
import os
import json
import string
import numpy as np
from collections import Counter
import spacy
import nltk

vocab_file = "data/vocab"
test_processed = "data/processed.txt"
glove_reduced = "data/glove_100d.txt"
char_vocab_file = "data/char_vocab.txt"


class Question(object):

    def __init__(self, question, a1, a2):
        self.question = question
        self.a1 = a1
        self.a2 = a2

    def __str__(self):
        print('QUESTION\n', self.question, '\n', self.a1, '\n', self.a2)
        return ""

    def generate(self):
        return [self.question, self.a1[0], self.a2[0], self.a1[1]]

    def get_len_q(self):
        return len(self.question)

    def get_len_c(self):
        return max(len(self.a1[0]), len(self.a1[1]))

class Text(object):

    def __init__(self, text):
        self.text = text
        self.questions = []

    def __iadd__(self, other):
        self.questions.append(other)
        return self

    def __str__(self):
        print('xxxTEXTxxx\n', self.text)
        for q in self.questions:
            print(q)
        return ""

    def get_len(self):
        return len(self.questions)

    def generate(self):
        for q in self.questions:

            yield [self.text] + q.generate()

    def get_max_lens(self):
        q_m = self.questions[0].get_len_q()
        c_m = self.questions[0].get_len_c()
        for item in self.questions:
            tmp_q = item.get_len_q()
            tmp_c = item.get_len_c()
            if tmp_c > c_m:
                c_m = tmp_c
            if tmp_q > q_m:
                q_m = tmp_q
        return len(self.text), q_m, c_m


class Corpus(object):

    def __init__(self):
        self.texts = []
        self.hatar = 0
        # nem biztos hogy ez kell
        self.vocab = []

    def __iadd__(self, other: Text):
        self.texts.append(other)
        return self

    def set_hatar(self):
        self.hatar = len(self.texts)

    def __str__(self):
        for t in self.texts:
            print(t)
        return ""

    def get_len(self):
        osszeg = 0
        for item in self.texts[:self.hatar]:
            osszeg += item.get_len()
        return osszeg

    def generate_train(self):
        for t in self.texts[:self.hatar]:
            generator = t.generate()

            while True:
                try:
                    yield next(generator)
                except StopIteration:
                    break

    def generate_dev(self):
        for t in self.texts[self.hatar:]:
            generator = t.generate()

            while True:
                try:
                    yield next(generator)
                except StopIteration:
                    break

    def get_max_lens(self):
        p_m, q_m, c_m = self.texts[0].get_max_lens()
        for item in self.texts:
            tmp_p, tmp_q, tmp_c = item.get_max_lens()
            if tmp_p > p_m:
                p_m = tmp_p
            if tmp_q > q_m:
                q_m = tmp_q
            if tmp_c > c_m:
                c_m = tmp_c
        return p_m, q_m, c_m



class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


nlp = spacy.load('en')


def make_vocab(record, is_nltk=True):
    if is_nltk:
        var = nltk.word_tokenize(record[0], )
        put_in_vocab(var)
        var = nltk.word_tokenize(record[1])
        put_in_vocab(var)
        var = nltk.word_tokenize(record[2][0])
        put_in_vocab(var)
        var = nltk.word_tokenize(record[3][0])
        put_in_vocab(var)
    else:
        var = nlp(record[0])
        var = [t.text.lower() for t in var]
        put_in_vocab(var)
        var = nlp(record[1])
        var = [t.text.lower() for t in var]
        put_in_vocab(var)
        var = nlp(record[2][0])
        var = [t.text.lower() for t in var]
        put_in_vocab(var)
        var = nlp(record[3][0])
        var = [t.text.lower() for t in var]
        put_in_vocab(var)


def put_in_vocab(tokens):
    for token in tokens:
        if token not in vocab:
            vocab[token] = 1
        else:
            vocab[token] += 1


def get_data(*args):
    corp = Corpus()
    for elem in args:

        print(elem)
        parser = ElementTree.parse(elem)
        instances = parser.findall('instance')

        for inst in instances:
            text = inst.find('text').text
            t = Text(text=text)
            questions = inst.find('questions').findall('question')

            for q in questions:
                data_answer = []
                ans = q.findall('answer')
                targ = ans[0].attrib['correct']
                if targ == "True":
                    targ = 1
                else:
                    targ = 0

                q = Question(question=q.attrib['text'], a1=(ans[0].attrib['text'], targ), a2=(ans[1].attrib['text'], 1-targ))
                t += q
            corp += t
        if corp.hatar == 0:
            corp.set_hatar()

    print("Reading Done")

    return corp


def read_vocab(file):
    with open(file, "r") as f:
        for line in f.readlines():
            vocab.add(line.strip())


def read_char_vocab(file):
    with open(file, "r") as f:
        for line in f.readlines():
            char_vocab.add(line.strip())


vocab, char_vocab = Dictionary(), Dictionary()
read_vocab(vocab_file)
read_char_vocab(char_vocab_file)


def get_tokenized_text(text):
    text_str = ""
    postag = []
    nlp = spacy.load('en')
    sentence = nlp(text)
    first = True
    for token in sentence:
        if first:
            first = False
        else:
            text_str += " "
        text_str += token.text
        postag.append(token.tag_)
    return text_str, postag


"""obsolete"""
def create_processed_data():
    read_vocab(vocab_file)
    with open(test_processed, "w"):
        pass
    corpus = get_data(train, dev)
    g = corpus.generate()
    counter = 0
    temp_text = ""
    tokenized_temp = ""
    passage_pos = []
    while True:
        try:
            if counter % 50 == 0:
                print(counter)
            item = next(g)
            record = dict()
            if item[0][:50] != temp_text[:50]:
                tokenized_temp, passage_pos = get_tokenized_text(item[0])

            record["text"] = tokenized_temp
            record["p_pos"] = passage_pos

            record["question"], record["q_pos"] = get_tokenized_text(item[1])
            record["answer_1"], record["c_1_pos"] = get_tokenized_text(item[2])
            record["answer_2"], record["c_2_pos"] = get_tokenized_text(item[3])
            record["label"] = item[-1]

            with open(test_processed, "a") as f:

                json.dump(record, f)
                f.write("\n")

            counter += 1
        except StopIteration:
            break


def reduced_glove():
    counter = 0
    with open(glove_reduced, "w") as f:
        pass
    from utils import vec
    for token in vocab.tokens():
        if counter % 750 == 0:
            print(counter)
        res = token
        for fl in vec(token):
            res += " " + str(fl)

        with open(glove_reduced, "a") as f:
            f.write(res)
            f.write('\n')
        counter += 1
    print(counter)


def vocab_to_low(v_file):
    words = []
    with open(v_file, "r") as f:
        for line in f.readlines():
            words.append(line.strip().lower())

    with open(v_file, "w") as f:
        for word in words:
            f.write(word)
            f.write('\n')



if __name__ == '__main__':
    train = "data/train-data.xml"
    dev = "data/dev-data.xml"
    # read_vocab(vocab_file)
    # vocab_to_low(vocab_file)
    # reduced_glove()
    # corpus = get_data(train, dev)
    # g = corpus.generate()
    #create_processed_data()

