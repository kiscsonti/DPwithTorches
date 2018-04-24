import nltk
from utils import make_vector_from_words
from utils import extend_array

"""Longest choice -> 23"""
def to_wordvec(text, question, a1, a2, sen_len, *args, **kwargs):

    global maxi
    words = nltk.word_tokenize(a1[0])
    a1_vecs = make_vector_from_words(words)
    a1_vecs = extend_array(sen_len, a1_vecs)

    words = nltk.word_tokenize(a2[0])
    a2_vecs = make_vector_from_words(words)
    a2_vecs = extend_array(sen_len, a2_vecs)

    return [a1_vecs, a2_vecs]



def get_target(text, question, a1, a2, *args, **kwargs):
    return [a1[1], a2[1]]
