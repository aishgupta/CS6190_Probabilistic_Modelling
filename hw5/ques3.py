import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import csv
from scipy.stats import truncnorm

TOLERANCE=1e-6

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

def get_data(filename):
	x, y = read_csv(filename)
	#x[:, :-1] = norm_data(x[:, :-1])
	y = y[:, np.newaxis]
	return(x,y)

def gibbs(x, y, burn_in, n_iter, w):
	accepted = np.array(w)
	for i in range(burn_in+n_iter):
		#update z
		w = w[:,None]
		wx = np.matmul(x, w)
		lower_limit = np.zeros(y.shape)
		upper_limit = np.zeros(y.shape)
		lower_limit[y[:,0]==0] = -1e6
		upper_limit[y[:,0]==1] = 1e6
		trunc_normal = truncnorm(lower_limit-wx, upper_limit-wx, loc = wx, scale = 1)
		z = trunc_normal.rvs((x.shape[0],1))
		#update w
		cov = np.linalg.inv(np.matmul(np.transpose(x), x) + np.eye(x.shape[1]))
		mu = (np.matmul(cov, np.matmul(np.transpose(x), z))).squeeze()
		w = multivariate_normal(mean=mu, cov=cov).rvs(1)

		if(i>=burn_in):
			accepted = np.vstack((accepted, w))
	return accepted, 1

def dU_dz(x, y, z):
	sigmoid = expit(np.matmul(x, z))
	#temp = sigmoid*(1-sigmoid)*(1-2*y)
	grad = (np.sum(x*(sigmoid-y), axis=0))[:,np.newaxis]+z
	return grad

def leapfrog(x, y, z, r, s, eps, L):

	for i in range(L):
		r -= (eps/2)*dU_dz(x, y, z)
		z += eps*np.matmul(np.linalg.inv(s), r)
		r -= (eps/2)*dU_dz(x, y, z)
	return (z, r)

def accept_prob(pos_dist, current_state, next_state, x, y, s):
	current_state_p = pos_dist(current_state, x, y, s)
	next_state_p = pos_dist(next_state, x, y, s)
	return(np.min([1, next_state_p/(current_state_p+TOLERANCE)]))

def total_energy(state, x, y, s):
	z = state[0]
	r = np.array(state[1])
	sigmoid = expit(np.matmul(x, z))
	u = -np.sum(np.log(sigmoid+TOLERANCE)*y + np.log(1-sigmoid+TOLERANCE)*(1-y)) + np.sum(z*z)*0.5
	k = 0.5*(np.matmul(np.matmul(r.transpose(),np.linalg.inv(s)), r)).squeeze()
	prob = np.exp(-u-k)
	return np.asscalar(prob)

def hybrid_monte_carlo(x, y, burn_in, n_iter, m, eps, L, z):
	s = np.eye(x.shape[1])
	r = multivariate_normal(mean=np.zeros(x.shape[1]), cov=s)
	z_p = z[:,np.newaxis]	#z_p.shape = (n,1)
	rejected = np.array(z_p)
	accepted = np.array(z_p)
	for i in range(1, burn_in + 1):
		r_p = r.rvs(1)[:, np.newaxis]  # sampling r from normal distribution
		z_n, r_n = leapfrog(x, y, np.copy(z_p), np.copy(r_p), s, eps, L)
		r_n *= (-1)
		prob = accept_prob(total_energy, [z_p, r_p], [z_n, r_n], x, y, s)
		u = np.random.uniform(0, 1, 1)
		if (u <= prob):
			z_p = z_n
	print("Burn-in for " + str(burn_in) + " iterations done!")

	for i in range(1, n_iter + 1):
		accept = False
		r_p = r.rvs(1)[:, np.newaxis]  # sampling r from normal distribution
		z_n, r_n = leapfrog(x, y, np.copy(z_p), np.copy(r_p), s, eps, L)
		r_n *= (-1)
		prob = accept_prob(total_energy, [z_p, r_p, s], [z_n, r_n, s], x, y, s)
		u = np.random.uniform(0, 1, 1)
		if (u <= prob):
			accept = True
		if (i % m == 0):
			if (accept):
				accepted = np.hstack((accepted, z_n))
			else:
				accepted = np.hstack((accepted, z_p))
				rejected = np.hstack((rejected, z_n))
		if (accept):
			z_p = z_n
	print(str(n_iter) + " iterations done!")
	return accepted.transpose(), 1 - ((rejected.shape[1]*1.0)/accepted.shape[1])

def get_samples(func_name, x, y, burn_in, n_iter, m, eps, L, z_init):
	if(func_name == gibbs):
		accepted, accept_rate = func_name(x, y, burn_in, n_iter//m, z_init)
	else:
		accepted, accept_rate = func_name(x, y, burn_in, n_iter, m, eps, L, z_init)
	return accepted, accept_rate

def predictive_accuracy(posterior_samples, x, y):
	x_repeat = np.repeat(x[np.newaxis,:,:], posterior_samples.shape[0], axis=0)		# x_repeat.shape = samples x N x d
	w = posterior_samples[:,:,np.newaxis]											# w.shape = samples x d x 1
	sigmoid = expit(np.matmul(x_repeat, w))											# sigmoid.shape = samples x N x 1
	y_true = np.repeat(y[np.newaxis, :, :], sigmoid.shape[0], axis=0)				# y_true.shape = samples x N x 1
	y_pred = np.zeros(y_true.shape)
	y_pred[sigmoid>0.5] = 1
	pred_acc = (np.sum(y_true == y_pred)+0.0)/(sigmoid.shape[0]*sigmoid.shape[1])
	return (pred_acc)

def predictive_log_likelihod(posterior_samples, x, y):
	x_repeat = x[:, np.newaxis, :]
	w = posterior_samples.transpose()
	sigmoid = expit(np.matmul(x_repeat, w)).squeeze()
	class_1 = ((np.sum(sigmoid, axis=1)/sigmoid.shape[1])+TOLERANCE)[:, np.newaxis]
	pred_like = np.sum(y*np.log(class_1) + (1-y)*np.log((1-class_1)))/class_1.shape[0]
	return (pred_like)

n = 5
z_init = np.zeros(n, dtype=np.float)	#z_init.shape = (n,)
n_iter = 10000
eps = 0.1; L = 20; m = 10
burn_in = 100000//5

train_filename = "../data/bank-note/train.csv"
test_filename = "../data/bank-note/test.csv"

x, y = get_data(train_filename)
print("Train Data\nX : ", x.shape,"\t Y : ", y.shape)	# X.shape :  (872, 5) 	 y.shape :  (872,1)
x_test, y_test = get_data(test_filename)
print("Test Data\nX : ", x_test.shape,"\t Y : ", y_test.shape)	# X.shape :  (500, 5) 	 y.shape :  (500,1)

print("=======================================================================")
print("                   GIBBS SAMPLING                         ")
print("=======================================================================")

w_init = np.ones((x.shape[1],))
posterior_samples, accept_rate  = get_samples(gibbs, x, y, burn_in, n_iter, m, eps, L, w_init)
pred_acc = predictive_accuracy(posterior_samples, x_test, y_test)
pred_like = predictive_log_likelihod(posterior_samples, x_test, y_test)
print("Predictive Accuracy: {} \tPredictive Likelihood: {} \t Acceptance Rate: {}".format(pred_acc, pred_like, accept_rate))

print("=======================================================================")
print("                   HYBRID MONTE CARLO SAMPLING                         ")
print("=======================================================================")

np.set_printoptions(precision=3)
eps_val = [0.1, 0.2, 0.5]
L_val = [10, 20, 50]
for eps in eps_val:
	for L in L_val:
		print("\n\nEpsilon = ", eps, "\tL = ", L)
		posterior_samples, accept_rate = get_samples(hybrid_monte_carlo, x, y, burn_in, n_iter, m, eps, L, z_init)
		pred_acc = predictive_accuracy(posterior_samples, x_test, y_test)
		pred_like = predictive_log_likelihod(posterior_samples, x_test, y_test)
