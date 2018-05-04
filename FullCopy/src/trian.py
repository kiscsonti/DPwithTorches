import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab, char_vocab

class TriAN(nn.Module):

    def __init__(self, args):
        super(TriAN, self).__init__()
        self.args = args
        self.embedding_dim = 100
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data[:2].normal_(0, 0.1)
        self.char_embedding = nn.Embedding(len(char_vocab), self.embedding_dim, padding_idx=0)
        self.char_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_p_emb_match = layers.SeqAttnMatch(self.embedding_dim)

        # Input size to RNN: word emb + question emb + pos emb + ner emb + manual features
        doc_input_size = 2 * self.embedding_dim + self.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN question encoder: word emb + pos emb
        qst_input_size = self.embedding_dim + self.embedding_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN answer encoder
        choice_input_size = 3 * self.embedding_dim + self.embedding_dim
        self.choice_rnn = layers.StackedBRNN(
            input_size=choice_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        choice_hidden_size = 2 * args.hidden_size

        # Answer merging
        self.c_self_attn = layers.LinearSeqAttn(choice_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(question_hidden_size)

        self.p_q_attn = layers.BilinearSeqAttn(x_size=doc_hidden_size, y_size=question_hidden_size)

        self.p_c_bilinear = nn.Linear(doc_hidden_size, choice_hidden_size)
        self.q_c_bilinear = nn.Linear(question_hidden_size, choice_hidden_size)
        #
        # char_input_size = self.embedding
        # self.char_rnn = layers.StackedBRNN(
        #     input_size=char_input_size,
        #     hidden_size=char_input_size,
        #     num_layers=1,
        #     dropout_rate=0,
        #     dropout_output=args.dropout_rnn_output,
        #     concat_layers=False,
        #     rnn_type=self.RNN_TYPES[args.rnn_type],
        #     padding=args.rnn_padding
        # )

    def forward(self, p, p_mask, q, q_mask, c, c_mask, d_chars, d_chars_mask, q_chars,
                q_chars_mask, c_chars, c_chars_mask):
        # print(d_chars)
        # print("passage", p, "\n", len(p), type(p))
        # print("passage pos", p_pos, "\n", len(p_pos), type(p_pos))
        # print("passage ner", p_ner, "\n", len(p_ner), type(p_ner))
        # print("passage mask", p_mask, "\n", len(p_mask), type(p_mask))
        # print("choice", c, "\n", len(c), type(c))
        # print("choice mask", c_mask, "\n", len(c_mask), type(c_mask))
        # print("question ", q, "\n", len(q), type(q))
        # print("question pos", q_pos, "\n", len(q_pos), type(q_pos))
        # print("question mask", q_mask, "\n", len(q_mask), type(q_mask))
        # print("f_tensor ", f_tensor, "\n", len(f_tensor), type(f_tensor))
        # print("passage question relation", p_q_relation, "\n", len(p_q_relation), type(p_q_relation))
        # print("passage choice relation", p_c_relation, "\n", len(p_c_relation), type(p_c_relation))
        p_emb, q_emb, c_emb = self.embedding(p), self.embedding(q), self.embedding(c)
        p_char_embed, q_char_embed, c_char_embed = self.char_embedding(d_chars), self.char_embedding(q_chars), self.char_embedding(c_chars)
        # p_char_hiddens = self.char_rnn(d_chars, d_chars_mask)
        # c_char_hiddens = self.choice_rnn(c_chars, c_chars_mask)
        # q_char_hiddens = self.question_rnn(q_chars, q_chars_mask)
        #TODO: create convolutional layer that works xD :)
        # d_chars, q_chars, c_chars =

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            p_emb = nn.functional.dropout(p_emb, p=self.args.dropout_emb, training=self.training)
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            c_emb = nn.functional.dropout(c_emb, p=self.args.dropout_emb, training=self.training)
            p_char_embed = nn.functional.dropout(p_char_embed, p=self.args.dropout_emb, training=self.training)
            q_char_embed = nn.functional.dropout(q_char_embed, p=self.args.dropout_emb, training=self.training)
            c_char_embed = nn.functional.dropout(c_char_embed, p=self.args.dropout_emb, training=self.training)

        p_q_weighted_emb = self.p_q_emb_match(p_emb, q_emb, q_mask)
        c_q_weighted_emb = self.c_q_emb_match(c_emb, q_emb, q_mask)
        c_p_weighted_emb = self.c_p_emb_match(c_emb, p_emb, p_mask)
        p_q_weighted_emb = nn.functional.dropout(p_q_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c_q_weighted_emb = nn.functional.dropout(c_q_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c_p_weighted_emb = nn.functional.dropout(c_p_weighted_emb, p=self.args.dropout_emb, training=self.training)
        # print('p_q_weighted_emb', p_q_weighted_emb.size())

        p_rnn_input = torch.cat([p_emb, p_q_weighted_emb, p_char_embed], dim=2)
        c_rnn_input = torch.cat([c_emb, c_q_weighted_emb, c_p_weighted_emb, c_char_embed], dim=2)
        q_rnn_input = torch.cat([q_emb, q_char_embed], dim=2)
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
