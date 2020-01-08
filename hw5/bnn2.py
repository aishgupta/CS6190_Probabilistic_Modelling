import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from hyperparameters import *

class Gaussian(nn.Module):
	def __init__(self, mu, rho):
		super().__init__()
		self.mu     = mu
		self.rho    = rho
		self.normal = torch.distributions.Normal(0, 1)

	@property
	def sigma(self):
		#return torch.sqrt(torch.log(1+torch.exp(self.rho))+TOLERANCE)
		return (torch.log(torch.exp(self.rho))+TOLERANCE)

	def sample(self):
		epsilon = self.normal.sample(self.rho.shape).type(self.mu.type()).to(device)
		return self.mu + self.sigma * epsilon

	def log_prob(self, input):
		return (-math.log(math.sqrt(2 * math.pi))
				- torch.log(self.sigma)
				- ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class StandardGaussian(nn.Module):
	def __init__(self, sigma):
		super().__init__()
		self.sigma = sigma
		self.gaussian = torch.distributions.Normal(0, self.sigma)

	def log_prob(self, input):
		return (self.gaussian.log_prob(input)).sum()

class BayesianLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features

		alpha = 1.0 / np.sqrt(self.in_features)

		# Weight parameters
		self.weight_mu  = nn.Parameter(torch.zeros(out_features, in_features))
		self.weight_rho = nn.Parameter(torch.ones(out_features, in_features)*alpha)
		self.weight     = Gaussian(self.weight_mu, self.weight_rho)
		# Bias parameters
		self.bias_mu    = nn.Parameter(torch.zeros(out_features))
		self.bias_rho   = nn.Parameter(torch.ones(out_features)*alpha)
		self.bias       = Gaussian(self.bias_mu, self.bias_rho)

		# Prior distributions
		self.weight_prior = StandardGaussian(1)#ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)#
		self.bias_prior = StandardGaussian(1)#ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)#StandardGaussian(1)
		self.log_prior = 0
		self.log_variational_posterior = 0

	def forward(self, input, sample=False, calculate_log_probs=False):
		if self.training or sample:
			weight = self.weight.sample()
			bias = self.bias.sample()
		else:
			weight = self.weight.mu
			bias = self.bias.mu
		if self.training or calculate_log_probs:
			self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
			self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
		else:
			self.log_prior, self.log_variational_posterior = 0, 0

		return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
	def __init__(self, n_hidden, activation):
		super().__init__()
		self.l1 = BayesianLinear(input_dim, n_hidden)
		self.l2 = BayesianLinear(n_hidden, n_hidden)
		self.l3 = BayesianLinear(n_hidden, n_class)
		if(activation=='relu'):
			self.activation_fn = nn.ReLU()
		else:
			self.activation_fn = nn.Tanh()

	def forward(self, x, sample=False):
		x = self.activation_fn(self.l1(x, sample))
		x = self.activation_fn(self.l2(x, sample))
		x = F.log_softmax(self.l3(x, sample), dim=1)
		return x

	def log_prior(self):
		return self.l1.log_prior \
			   + self.l2.log_prior \
			   + self.l3.log_prior

	def log_variational_posterior(self):
		return self.l1.log_variational_posterior \
			   + self.l2.log_variational_posterior \
			   + self.l3.log_variational_posterior

	def sample_elbo(self, input, target, samples=SAMPLES):

		outputs = torch.zeros(samples, input.shape[0], n_class).to(device)
		log_priors = torch.zeros(samples).to(device)
		log_variational_posteriors = torch.zeros(samples).to(device)
		for i in range(samples):
			outputs[i] = self.forward(input, sample=True)
			log_priors[i] = self.log_prior()
			log_variational_posteriors[i] = self.log_variational_posterior()
		log_prior = log_priors.mean()
		log_variational_posterior = log_variational_posteriors.mean()
		outputs = outputs.mean(dim= 0)
		neg_log_like = F.nll_loss(outputs, target, reduction='mean')
		loss = neg_log_like #+ (log_variational_posterior-log_prior)
		pred = outputs.argmax(dim= 1)
		train_acc = ((pred.eq(target.view_as(pred)).sum())*100.0)/target.shape[0]
		return loss, train_acc

def train(net, train_loader, optimizer, epoch):
	net.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target   = data.to(device), target.to(device)
		net.zero_grad()
		loss, train_acc = net.sample_elbo(data, target, samples=SAMPLES)
		if(epoch%log_iter==0):
			print("Epoch : {:4d} \t Training Loss : {:6.4f} \t Trainig Accuracy : {:4.2f}%".format(epoch, loss, train_acc))
		loss.backward()
		optimizer.step()

def predictive_accuracy(net, test_loader):
	net.eval()
	with torch.no_grad():
		test_acc = 0
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target   = data.to(device), target.to(device)
			outputs = torch.zeros(TEST_SAMPLES, data.shape[0], n_class).to(device)
			for i in range(TEST_SAMPLES):
				outputs[i] = net(data, sample=True)
				pred = outputs[i].argmax(dim=1)
				test_acc += ((pred.eq(target.view_as(pred)).sum()) * 100.0) / target.shape[0]
				#print(test_acc)
			#outputs = outputs.mean(0)
			#pred = outputs.argmax(dim=1)
			#test_acc = (torch.sum(pred == target) * 100.0) / target.shape[0]
		test_acc = test_acc/TEST_SAMPLES
	return test_acc

def predictive_log_likelihood(net, test_loader):
	net.eval()
	with torch.no_grad():
		pred_like = 0
		counter = 0

		for batch_idx, (data, target) in enumerate(test_loader):
			#print(target.shape)
			data, target   = data.to(device), target.float().to(device)
			outputs = torch.zeros(TEST_SAMPLES, data.shape[0], n_class).to(device)
			for i in range(TEST_SAMPLES):
				outputs[i] = net(data, sample=True)
			output = torch.sum(outputs, axis = 0)/TEST_SAMPLES
			pred_like += torch.mean(target*output[:,1] + (1-target)*output[:,0])
			counter+=1
		pred_like = pred_like/counter
	return pred_like