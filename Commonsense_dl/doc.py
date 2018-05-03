import torch
import numpy as np


class Example:

    def __init__(self, input_dict):
        self.passage = input_dict['p_words']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.label = input_dict['label']

        self.d_tensor = torch.LongTensor([vocab[w] for w in self.passage.split()])
        self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
        self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer_1: %s\n Answer_1: %s, Label: %d' % (self.passage, self.question, self.choice1, self.choice2, self.label)

    def get_record(self):
        return self.d_tensor, self.q_tensor, self.c_tensor
