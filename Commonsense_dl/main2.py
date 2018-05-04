import os
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import LSTMClassifier
from preprocess import get_data, vocab
from utils import load_data
import sys

epochs = 6
batch_size = 32
learning_rate = 0.01
train = "data/train-data.xml"
dev = "data/dev-data.xml"


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
    corpus = get_data(train, dev)

    ### create model
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, label_size=n_label,
                           batch_size=batch_size, use_gpu=False)

    ### data processing
    train_data = load_data(train)
    dev_data = load_data(dev)

    # TODO: try out other optimizers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        batch_num = len(train_data) / batch_size
        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for i in range(batch_num):
            train_inputs = train_data[i*batch_size: (i+1)*batch_size]
            #TODO: itt hagytam abba
            #test_labels = [ for l in train_data[i*batch_size: (i+1)*batch_size]]

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

    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = sentence_len

    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param

    if use_plot:
        import PlotFigure as PF

        PF.PlotFigure(result, use_save)
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)
