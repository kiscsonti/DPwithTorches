import torch
import numpy as np

from utils import vocab, pos_vocab, ner_vocab, rel_vocab, char_vocab, text_to_grams, text_to_char_index


class Example:

    def __init__(self, input_dict):
        self.id = input_dict['id']
        self.passage = input_dict['d_words']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.label = input_dict['label']

        self.d_tensor = torch.LongTensor([vocab[w] for w in self.passage.split()])
        self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
        self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])
        # self.d_char = torch.LongTensor(text_to_grams(text_to_char_index(self.passage)))
        # self.q_char = torch.LongTensor(text_to_grams(text_to_char_index(self.question)))
        # self.c_char = torch.LongTensor(text_to_grams(text_to_char_index(self.choice)))
        # self.d_char = torch.LongTensor(text_to_char_index(self.passage))
        # self.q_char = torch.LongTensor(text_to_char_index(self.question))
        # self.c_char = torch.LongTensor(text_to_char_index(self.choice))
        self.d_char = torch.LongTensor([char_vocab[w] for w in self.passage.split()])
        self.q_char = torch.LongTensor([char_vocab[w] for w in self.question.split()])
        self.c_char = torch.LongTensor([char_vocab[w] for w in self.choice.split()])

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer: %s, Label: %d' % (self.passage,
                                                                       self.question,
                                                                       self.choice,
                                                                       self.label,
                                                                       )


def _to_indices_and_mask(batch_tensor, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0)
    if need_mask:
        return indices, mask
    else:
        return indices


def _to_feature_tensor(features):
    mx_len = max([f.size(0) for f in features])
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :].copy_(f)
    return f_tensor


def batchify(batch_data):
    p, p_mask = _to_indices_and_mask([ex.d_tensor for ex in batch_data])
    q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
    c, c_mask = _to_indices_and_mask([ex.c_tensor for ex in batch_data])
    # d_chars = ([ex.d_char for ex in batch_data])
    # q_chars = ([ex.q_char for ex in batch_data])
    # c_chars = ([ex.c_char for ex in batch_data])
    d_chars, d_chars_mask = _to_indices_and_mask([ex.d_char for ex in batch_data])
    q_chars, q_chars_mask = _to_indices_and_mask([ex.q_char for ex in batch_data])
    c_chars, c_chars_mask = _to_indices_and_mask([ex.c_char for ex in batch_data])

    y = torch.FloatTensor([ex.label for ex in batch_data])
    return p, p_mask, q, q_mask, c, c_mask, d_chars, d_chars_mask, q_chars, q_chars_mask, c_chars, c_chars_mask, y
