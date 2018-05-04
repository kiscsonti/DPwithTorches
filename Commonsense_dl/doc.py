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
        return 'Passage: %s\n Question: %s\n Answer_1: %s, Label: %d' % (self.passage,
                                                                         self.question,
                                                                         self.choice,
                                                                         self.label)

    # def get_record(self):
    #     return self.d_tensor, self.q_tensor, self.c_tensor


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


def batchify(batch_data):
    p, p_mask = _to_indices_and_mask([ex.d_tensor for ex in batch_data])
    q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
    c, c_mask = _to_indices_and_mask([ex.c_tensor for ex in batch_data])
    y = torch.FloatTensor([ex.label for ex in batch_data])
    return p, p_mask, q, q_mask, c, c_mask, y
