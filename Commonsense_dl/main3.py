import os
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import LSTMClassifierNoEmbedding
from preprocess import get_data
import sys


epochs = 12
batch_size = 32
learning_rate = 0.01
train = "train-data.xml"
dev = "dev-data.xml"

if __name__ == '__main__':
    ### parameter setting
    input = 300
    hidden_dim = 128
    sentence_len = 32
    n_label = 2
    corpus = get_data(train, dev)


    ### create model
    model = LSTMClassifierNoEmbedding(input_dim=input,
                                      hidden_dim=hidden_dim,
                                      label_size=n_label,
                                      batch_size=batch_size)

    ### data processing


    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        batch_num = corpus.get_len_train()/batch_size

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for i in range(batch_num):



            train_inputs = Variable(train_inputs)

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()

            output = model(train_inputs.t())

            sys.exit(0)
            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.data[0]

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, testdata in enumerate(test_loader):
            test_inputs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else:
                test_inputs = Variable(test_inputs)

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t())

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.data[0]
        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))
