#!/usr/bin/python3

import numpy as np
from scipy import sparse
from math import *
import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pyvacy import optim as DP_optim
# import seaborn as sns

class LSTM(nn.Module):

    def __init__(self, text_field, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)
        # self.fc = nn.Linear(dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def LSTM_train(model,
          optimizer,
          train_loader,
          valid_loader,
          file_path,
          device,
          num_epochs = 5,
          criterion = nn.BCELoss(),
          best_valid_loss = float("Inf")):
    
    # eval_every = floor(len(train_loader)/10)
    eval_every = 10
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, (text, text_len)), _ in train_loader:           
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for (labels, (text, text_len)), _ in valid_loader:
                      labels = labels.to(device)
                      text = text.to(device)
                      text_len = text_len.to(device)
                      output = model(text, text_len)

                      loss = criterion(output, labels)
                      valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

def LSTM_evaluate(model, test_loader, device, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (text, text_len)), _ in test_loader:           
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    acc = accuracy_score(y_true, y_pred)
    print(acc)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    # cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    # ax.set_title('Confusion Matrix')

    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')

    # ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    # ax.yaxis.set_ticklabels(['FAKE', 'REAL'])

def LSTM_classification(data_train, data_test):

	# print(NAttackHam, NAttackSpam)
	# data_train = 'Attack_{}_{}_train.csv'.format(NAttackHam, NAttackSpam)
	# data_test = 'test.csv'
	training_parameters = {
	'l2_norm_clip': 2.0,
	'noise_multiplier': 0.1,
	'batch_size': 2048,
	}
	batch = 512 # 512
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Fields
	label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
	text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
	fields = [('label', label_field), ('text', text_field)]

	# TabularDataset
	train, test = TabularDataset.splits(path='dataset/full', train=data_train, test=data_test,
											format='CSV', fields=fields, skip_header=True)
	# Iterators
	train_iter = BucketIterator(train, batch_size=batch, sort_key=lambda x: len(x.text),
								device=device, sort=True, sort_within_batch=True)
	# valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text),
								# device=device, sort=True, sort_within_batch=True)
	test_iter = BucketIterator(test, batch_size=512, sort_key=lambda x: len(x.text),
								device=device, sort=True, sort_within_batch=True)
	# Vocabulary
	text_field.build_vocab(train, min_freq=3)

	model = LSTM(text_field=text_field).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.05)
	# optimizer = optim.SGD(model.parameters(), lr=0.05)
	# optimizer = DP_optim.DPSGD(params=model.parameters(), **training_parameters, lr=0.05)
	
	LSTM_train(model=model, optimizer=optimizer, num_epochs=10, train_loader=train_iter, valid_loader=test_iter, file_path='LSTM', device=device)
	
	# load_checkpoint('LSTM' + '/model.pt', model, optimizer, device=device)
	LSTM_evaluate(model, test_iter, device=device)

	return

NAttackHam = 2000
NAttackSpam = 4000
# LSTM_classification('Attack_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), 'test.csv')
# LSTM_classification('Sanitized_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), 'test.csv')
# LSTM_classification('Sanitized_KNN_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), 'test.csv')
attack = [(0,0), (0,1000), (0,2000), (0,3000), (0,4000), (0,5000), (0,6000), (1000,1000), (1000,2000), (2000,1000), (2000,2000), (2000,4000), (4000,2000)]
for NAttackHam, NAttackSpam in attack:
	print(NAttackHam, NAttackSpam)
	LSTM_classification('Sanitized_KNN_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), 'test.csv')
	# LSTM_classification('Sanitized_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), 'test.csv')