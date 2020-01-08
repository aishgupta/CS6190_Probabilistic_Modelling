import csv
import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import argparse

parser = argparse.ArgumentParser(description='Logistic / Probit model')
parser.add_argument('--method', type=str, default = 'logistic', help='logistic/probit')
parser.add_argument('--initialise', type=str, default = 'random', help='random/zeros/xavier')
parser.add_argument('--convergence', type =str, default = 'bfgs', help = 'bfgs/hessian')
parser.add_argument('--data_dir', type = str, default = '../bank-note/', help = 'Path of data directory')
args = parser.parse_args()

path1 = args.data_dir

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

	print("Data read !!!\n X : ", x.shape,"\t Y : ", y.shape)	# X.shape :  (872, 5) 	 y.shape :  (872,)
	return (x,y)

def initialise_w(initialise):

	if(initialise == 'random'):
		w = np.random.randn(d,1)
		print("w is initialised from N[0,1]")
	elif(initialise == 'zeros'):
		w = np.zeros((d,1))
		print("w is initialised as a zero vector")
	elif(initialise == 'xavier'):
		w = np.random.randn(d,1)/np.sqrt(d)
		print("w is initialised using xavier method")
	else:
		print("Method unknown")
	return w

def compute_mu(X, w, algo):

	if(algo == 'logistic'):
		mu = expit(np.matmul(X,w))
	
	elif(algo == 'probit'):
		phi=np.matmul(X,w)
		mu = norm.cdf(phi)

	else:
		print("Unsupported method name")
	mu = mu.reshape(X.shape[0],1)
	return mu

def neg_log_posterior(w):

	w=w.reshape(-1,1)
	epsilon = 1e-9
	mu = compute_mu(X, w, method)		# y.shape =  (872, 1) mu.shape =  (872, 1)
	prob_1 = Y*np.log(mu+epsilon)
	prob_0 = (1-Y)*np.log(1-mu+epsilon)
	log_like = np.sum(prob_1) + np.sum(prob_0)
	w_norm = np.power(np.linalg.norm(w),2)
	neg_log_pos = -log_like+w_norm/2
	print("neg_log_posterior = {:.4f} \tlog_like = {:.4f} \tw_norm = {:.4f}".format(neg_log_pos, log_like, w_norm))
	return(neg_log_pos)

def first_derivative(w):
		
	mu = compute_mu(X, w, method)
	epsilon = 1e-9
	#w = w.reshape(d,1)
	if(method == 'probit'):
		phi=np.matmul(X,w)
		grad_mu = X*(scipy.stats.norm.pdf(phi,0,1).reshape(-1,1))		# grad_mu.shape (872, 5)
		
		#np.sum((y*(1/(mu)) - (1-y)*(1/(1-mu)))*grad_mu,0).shape = (5,)
		#w.shape = (5,)
		return(np.sum((- Y*(1/(mu)) + (1-Y)*(1/(1+epsilon-mu)))*grad_mu,0) + w).squeeze()	

	elif(method == 'logistic'):
		grad = np.matmul(np.transpose(X), (mu-Y)) + w.reshape(d,1)
		grad = grad.squeeze()
		#print("grad",grad, grad.shape, np.matmul(np.transpose(X), (mu-y)).shape, w.shape)
		return(grad)

def second_deivative(w,X,y):

	n,d = X.shape
	mu = compute_mu(X, w, method)		#softmax of Xw
	R = np.eye(n)
	if(method == 'logistic'):
		for i in range(n):
			R[i,i] = mu[i,0] * (1-mu[i,0])
		
	else:
		phi=np.matmul(X,w)
		for i in range(n):
			t1 = (y[i] - mu[i,0])/(mu[i,0] * (1-mu[i,0]))
			t2 = scipy.stats.norm.pdf(phi[i,0],0,1)
			t3 = (1-y[i])/np.power(1-mu[i,0],2) + y[i]/np.power(mu[i,0],2)
			R[i,i] = t1*t2*np.matmul(X[i],w) + t3*t2*t2

	return(np.matmul(np.matmul(np.transpose(X),R),X) + np.eye(d))		#returns Hessian matrix

def test(w, X, y):

	n,d = X.shape
	mu = compute_mu(X, w, method)
	#print(mu.shape, n, d)
	yhat = np.zeros((n,1)).astype(np.float)
	yhat[mu>0.5]=1
	correct = np.sum(yhat==y)
	return(correct,n)

def train(initialise = 'random'):

	np.random.seed(19)
	w = initialise_w(initialise)			
	for j in range(max_iter):

		grad1 = first_derivative(w.squeeze()).reshape(d,1)	# grad1.shape = (5,1)
		H = second_deivative(w, X, Y)		# H.shape = (5,5)
		delta_w = np.matmul(np.linalg.inv(H),grad1)		# delta_w.shape = (5,5)
		w = w - delta_w		# w.shape = (5,1)	
		diff = np.linalg.norm(delta_w)

		correct,n = test(w, testX, testY)
		print("Iteration : {0:4d} \t \t Correct : {1:4d} \t Accuracy : {2:5.4f} \t Delta_w_norm : {3:8.6f}".format(j,correct,float(correct)/n, diff))

		if(diff < tolerance):
			print("tolerance reached at the iteration : ",j)
			break
	print("_____________Model trained______________")
	print("Model weights : ", np.transpose(w))

def train_bfgs(initialise = 'random'):

	#seed=9 91% test accuracy
	#seed=0 96% test accuracy
	#seed=11 97% test accuracy
	w = initialise_w(initialise)
	res = minimize(neg_log_posterior, w, method='BFGS', jac=first_derivative)
	correct,n = test(res.x, testX, testY)
	print("\n_____________Model trained______________\n")
	print("\nModel weights : ", res.x)
	print("\n_____________Test Accuracy______________\n")
	
	print("Correct : {0:4d} \t Accuracy : {1:4.2f} ".format(correct,float(correct)/n))

X, Y = read_csv(path1+'train.csv')
testX, testY = read_csv(path1+'test.csv')

np.random.seed(0)

n,d = X.shape
n1,d1 = testX.shape

Y = Y.reshape(n,1)
testY = testY.reshape(n1,1)	
max_iter = 100
tolerance = 0.00001
#method='probit'
method = args.method
initialise = args.initialise
if(args.convergence == 'hessian'):
	train(initialise)
elif(args.convergence == 'bfgs'):
	train_bfgs(initialise)
else:
	print("Unknown value .")


