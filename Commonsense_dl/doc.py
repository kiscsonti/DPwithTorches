import torch
import numpy as np


class Example:

    def __init__(self, input_dict):
        self.passage = input_dict['text']
        self.question = input_dict['question']
        self.choice1 = input_dict['answer_1']
        self.choice2 = input_dict['answer_2']
        self.label = input_dict['label']

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer_1: %s\n Answer_1: %s, Label: %d' % (self.passage, self.question, self.choice1, self.choice2, self.label)

