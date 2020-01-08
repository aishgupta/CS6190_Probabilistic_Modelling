import torch
import torch.nn.functional as F
import numpy as np

from hyperparameters import *
from bank_data import *
from bnn2 import *
import os
import matplotlib.pyplot as plt

path = os.getcwd()
path = path+"/images"

train_filename= "../data/bank-note/train.csv"
test_filename = "../data/bank-note/test.csv"

train_data    = BankData(train_filename)
train_loader  = torch.utils.data.DataLoader(dataset= train_data,  batch_size= train_batch_size, shuffle= True , num_workers = workers)

test_data     = BankData(test_filename)
test_loader   = torch.utils.data.DataLoader(dataset= test_data,  batch_size= test_batch_size, shuffle= False, num_workers = workers)

n_hidden_val = [10, 20, 50]
#lr_val       = [0.001, 0.0005, 0.0001, 0.00001]
activation_val = ['relu', 'tanh']
for activation in activation_val:
	for n_hidden in n_hidden_val:
		#for lr in lr_val:
		net = BayesianNetwork(n_hidden= n_hidden, activation= activation).to(device)
		optimizer = torch.optim.Adam(net.parameters(), lr = lr)
		print("\n\n==============================================================================\n")
		print("Activation Function : {} \tHidden Layer Size : {} \tLearning rate : {}\n".format(activation, n_hidden, lr))
		train_like = []
		test_like = []
		for epoch in range(1,max_epoch+1):
			train(net, train_loader, optimizer, epoch)
			train_like.append(predictive_log_likelihood(net, train_loader))
			test_like.append(predictive_log_likelihood(net, test_loader))
		pred_acc = predictive_accuracy(net, test_loader)
		pred_like = predictive_log_likelihood(net, test_loader)
		print("\n Predictive Accuracy : {:.4f} \tPredictive Log Likelihood : {:.4f}".format(pred_acc, pred_like))

		plt.plot(np.arange(max_epoch)+1, train_like)
		plt.xlabel("Epoch")
		plt.ylabel("Predictive log-likelihood")
		plt.grid()
		plt.savefig(path + "/Train_predictive_log_like.png")
		plt.clf()
		plt.plot(np.arange(max_epoch) + 1, test_like)
		plt.xlabel("Epoch")
		plt.ylabel("Predictive log-likelihood")
		plt.grid()
		plt.savefig(path + "/Test_predictive_log_like.png")