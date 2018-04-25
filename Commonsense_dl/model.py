import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import logging
import sys
from preprocess import vocab
from utils import glove_data_file

logger = logging.getLogger()

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.load_embeddings(vocab.tokens(), glove_data_file)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        sys.exit(0)
    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y


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
