
from xml.etree import ElementTree
import nltk
import pandas as pd
import numpy as np
import spacy


class Question(object):

    def __init__(self, question, a1, a2):
        self.question = question
        self.a1 = a1
        self.a2 = a2

    def __str__(self):
        print('QUESTION\n', self.question, '\n', self.a1, '\n', self.a2)
        return ""

    def get_feature(self, method, text, args, **kwargs):
        return method(text, self.question, self.a1, self.a2, args)

    def generate(self):
        return [self.question, self.a1, self.a2]


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

    def get_feature(self, method, args, **kwargs):
        res = []
        for q in self.questions:

            res.extend(q.get_feature(method, self.text, args))
        return res

    def generate(self):
        for q in self.questions:
            yield [self.text] + q.generate()


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
        self.hatar =     len(self.texts)

    def __str__(self):
        for t in self.texts:
            print(t)
        return ""

    def get_len_train(self):
        osszeg = 0
        for item in self.texts[:self.hatar]:
            osszeg += item.get_len()
        return osszeg

    def get_len_test(self):
        osszeg = 0
        for item in self.texts[self.hatar:]:
            osszeg += item.get_len()
        return osszeg

    def get_feature(self, method, args, **kwargs):
        res = []
        i =0
        for text in self.texts:
            print(i, "text danzo")
            res.extend(text.get_feature(method, args))
            i += 1
        return res

    def generate(self):
        for text in self.texts:
            generator = text.generate()
            while True:
                try:
                    yield next(generator)
                except StopIteration:
                    break




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


vocab = dict()
def put_in_vocab(tokens):
    for token in tokens:
        if token not in vocab:
            vocab[token] = 1
        else:
            vocab[token] += 1

def load_vocab(filename):
    with open(filename, "r") as f:
        for item in f.readlines():
            vocab[item] = len(vocab)

