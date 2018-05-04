import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import logging
import sys
from preprocess import vocab
from utils import glove_data_file
import numpy as np
import layers

logger = logging.getLogger()


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, use_gpu, dropout_emb):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.dropout_emb = dropout_emb

        self.word_embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.load_embeddings(vocab.tokens(), glove_data_file)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

        self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_p_emb_match = layers.SeqAttnMatch(self.embedding_dim)

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, p, p_mask, q, q_mask, c, c_mask, y):
        p_emb, q_emb, c_emb = self.embedding(p), self.embedding(q), self.embedding(c)

        # Dropout on embeddings
        if self.dropout_emb > 0:
            p_emb = nn.functional.dropout(p_emb, p=self.dropout_emb, training=self.training)
            q_emb = nn.functional.dropout(q_emb, p=self.dropout_emb, training=self.training)
            c_emb = nn.functional.dropout(c_emb, p=self.dropout_emb, training=self.training)

        p_q_weighted_emb = self.p_q_emb_match(p_emb, q_emb, q_mask)
        c_q_weighted_emb = self.c_q_emb_match(c_emb, q_emb, q_mask)
        c_p_weighted_emb = self.c_p_emb_match(c_emb, p_emb, p_mask)
        p_q_weighted_emb = nn.functional.dropout(p_q_weighted_emb, p=self.dropout_emb, training=self.training)
        c_q_weighted_emb = nn.functional.dropout(c_q_weighted_emb, p=self.dropout_emb, training=self.training)
        c_p_weighted_emb = nn.functional.dropout(c_p_weighted_emb, p=self.dropout_emb, training=self.training)
        # print('p_q_weighted_emb', p_q_weighted_emb.size())

        p_rnn_input = torch.cat([p_emb, p_q_weighted_emb, p_pos_emb, p_ner_emb, f_tensor, p_q_rel_emb, p_c_rel_emb], dim=2)
        c_rnn_input = torch.cat([c_emb, c_q_weighted_emb, c_p_weighted_emb], dim=2)
        q_rnn_input = torch.cat([q_emb, q_pos_emb], dim=2)
        # print('p_rnn_input', p_rnn_input.size())

        p_hiddens = self.doc_rnn(p_rnn_input, p_mask)
        c_hiddens = self.choice_rnn(c_rnn_input, c_mask)
        q_hiddens = self.question_rnn(q_rnn_input, q_mask)
        # print('p_hiddens', p_hiddens.size())

        q_merge_weights = self.q_self_attn(q_hiddens, q_mask)
        q_hidden = layers.weighted_avg(q_hiddens, q_merge_weights)

        p_merge_weights = self.p_q_attn(p_hiddens, q_hidden, p_mask)
        # [batch_size, 2*hidden_size]
        p_hidden = layers.weighted_avg(p_hiddens, p_merge_weights)
        # print('p_hidden', p_hidden.size())

        c_merge_weights = self.c_self_attn(c_hiddens, c_mask)
        # [batch_size, 2*hidden_size]
        c_hidden = layers.weighted_avg(c_hiddens, c_merge_weights)
        # print('c_hidden', c_hidden.size())

        logits = torch.sum(self.p_c_bilinear(p_hidden) * c_hidden, dim=-1)
        logits += torch.sum(self.q_c_bilinear(q_hidden) * c_hidden, dim=-1)
        proba = F.sigmoid(logits)
        # print('proba', proba.size())

        return proba

    """
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y
    """

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.word_embeddings.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[vocab[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))


    def evaluate(self, dev_data, debug=False, eval_train=False):
        if len(dev_data) == 0:
            return -1.0
        self.eval()
        correct, total, prediction, gold = 0, 0, [], []
        #dev_data = sorted(dev_data, key=lambda ex: ex.id)
        for batch_input in self._iter_data(dev_data):
            feed_input = [x for x in batch_input[:-1]]
            y = batch_input[-1].data.cpu().numpy()
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
            gold += [int(label) for label in y]
            assert(len(prediction) == len(gold))

        if eval_train:
            prediction = [1 if p > 0.5 else 0 for p in prediction]
            acc = sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, gold)]) / len(gold)
            return acc

        cur_pred, cur_gold, cur_choices = [], [], []
        if debug:
            writer = open('./data/output.log', 'w', encoding='utf-8')
        for i, ex in enumerate(dev_data):
            if i + 1 == len(dev_data):
                cur_pred.append(prediction[i])
                cur_gold.append(gold[i])
                cur_choices.append(ex.choice)
            if (i > 0 and ex.id[:-1] != dev_data[i - 1].id[:-1]) or (i + 1 == len(dev_data)):
                py, gy = np.argmax(cur_pred), np.argmax(cur_gold)
                if debug:
                    writer.write('Passage: %s\n' % dev_data[i - 1].passage)
                    writer.write('Question: %s\n' % dev_data[i - 1].question)
                    for idx, choice in enumerate(cur_choices):
                        writer.write('*' if idx == gy else ' ')
                        writer.write('%s  %f\n' % (choice, cur_pred[idx]))
                    writer.write('\n')
                if py == gy:
                    correct += 1
                total += 1
                cur_pred, cur_gold, cur_choices = [], [], []
            cur_pred.append(prediction[i])
            cur_gold.append(gold[i])
            cur_choices.append(ex.choice)

        acc = 1.0 * correct / total
        if debug:
            writer.write('Accuracy: %f\n' % acc)
            writer.close()
        return acc



class LSTMClassifierNoEmbedding(nn.Module):

    def __init__(self, input_dim, hidden_dim, label_size, batch_size):
        super(LSTMClassifierNoEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        return y
