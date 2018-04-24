import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

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
        print("Embeds: ", embeds)
        x = embeds.view(len(sentence), self.batch_size, -1)
        print("\n\n\n")
        print("Xs: ", x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y


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
