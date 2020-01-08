import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import csv

def read_csv(filename):
	x=[]
	y=[]
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for i in csv_reader :
			x.append(i[0:-1])
			y.append(i[-1])
	x = np.array(x).astype(np.float)
	x = np.hstack((x,np.ones((x.shape[0],1))))		#for the bias term
	y = np.array(y).astype(np.float)
	return (x,y)

def norm_data(x):
	#To normalise the data within [-1, 1]
	x =  (x - np.mean(x, axis=0))*(1/(np.max(x,axis=0) - np.min(x, axis=0)))    #max(x) = 0.47, min(x) = -0.53
	return(x)

class BankData(Dataset):

	def __init__(self, filename):
		x, y = read_csv(filename)
		#x[:, :-1] = norm_data(x[:, :-1])
		self.X, self.Y = x, y

	def __getitem__(self, index):

		x = torch.tensor(self.X[index]).float()
		y = torch.tensor(self.Y[index]).long()
		return (x, y)

	def __len__(self):
		return self.X.shape[0]