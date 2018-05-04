import os
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import LSTMClassifier
from preprocess import get_data
from utils import load_data
from doc import batchify
import sys
import time
import numpy as np

epochs = 6
batch_size = 32
learning_rate = 0.01
train = "data/train-data.xml"
dev = "data/dev-data.xml"

processed_train = "data/my_processed_train.json"
processed_dev = "data/my_processed_dev.json"

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    ### parameter setting
    input = 100
    hidden_dim = 128
    embedding_dim = 100
    sentence_len = 32
    n_label = 2
    dropout_emb = 0.4
    # corpus = get_data(train, dev)

    ### create model
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, label_size=n_label,
                           batch_size=batch_size, use_gpu=False, dropout_emb=dropout_emb)

    ### data processing
    train_data = load_data(processed_train)
    dev_data = load_data(processed_dev)

    num_iter = (len(train_data) + batch_size - 1) // batch_size



    for i in range(num_iter):
        p, p_mask, q, q_mask, c, c_mask, y = batchify(train_data[i*batch_size: (i+1)*batch_size])
        # print("passage", p, "\n", len(p), type(p))
        # print("passage mask", p_mask, "\n", len(p_mask), type(p_mask))
        # print("choice", c, "\n", len(c), type(c))
        # print("choice mask", c_mask, "\n", len(c_mask), type(c_mask))
        # print("question ", q, "\n", len(q), type(q))
        # print("question mask", q_mask, "\n", len(q_mask), type(q_mask))

        break



    for i in range(epochs):
        print('Epoch %d...' % i)
        if i == 0:
            dev_acc = model.evaluate(dev_data)
            print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        np.random.shuffle(train_data)
        cur_train_data = train_data
        #####################x

        iter_cnt, num_iter = 0, (len(train_data) + batch_size - 1) // batch_size
        for j in range(num_iter):
            batch_input = batchify(train_data[j*batch_size: (j+1)*batch_size])
            feed_input = [x for x in batch_input[:-1]]
            y = batch_input[-1]
            print(feed_input)
            import sys

            sys.exit(0)
            pred_proba = self.network(*feed_input)

            loss = F.binary_cross_entropy(pred_proba, y)
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 20 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.data[0]))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])










        train_acc = model.evaluate(train_data[:2000], debug=False, eval_train=True)
        print('Train accuracy: %f' % train_acc)
        dev_acc = model.evaluate(dev_data, debug=True)
        print('Dev accuracy: %f' % dev_acc)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    print('Best dev accuracy: %f' % best_dev_acc)
