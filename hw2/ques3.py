import csv
import numpy as np
import scipy.stats
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import argparse

parser = argparse.ArgumentParser(description='multi-class logistic/ one-vs-all logistic')
parser.add_argument('--initialise', type=str, default = 'xavier', help='xavier/zeros')
parser.add_argument('--num_class', type =str, default = 'multiclass', help = 'multiclass/oneclass')
parser.add_argument('--data_dir', type = str, default = '../car/', help = 'Path of data directory')
args = parser.parse_args()

path = args.data_dir

def read_csv(filename):

	x=[]
	y=[]	
	
	attr_map =[{"vhigh":0, "high":1, "med":2, "low":3},
	{"vhigh":0, "high":1, "med":2, "low":3}, 
	{"2":0, "3":1, "4":2, "5more":3},
	{"2":0, "4":1, "more":2},
	{"small":0, "med":1, "big":2},
	{"low":0, "med":1, "high":2},
	{"unacc":0, "acc":1, "good":2, "vgood":3}]

	attr_val=[0,4,8,12,15,18]
	x_len=21
	y_len=4

	with open(filename) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    for i in csv_reader :
	    	t1=np.zeros((x_len,1))
	    	t2=np.zeros((y_len,1))
	    	for j in range(len(i)-1):
	    		t1[attr_val[j]+attr_map[j][i[j]]]=1
	    		t2[attr_map[-1][i[-1]]]=1
	    	t1=t1.reshape(x_len)
	    	t2=t2.reshape(y_len)
	    	x.append(t1)
	    	y.append(t2)

	x = np.array(x)
	x = np.hstack((x, np.ones((x.shape[0],1)))).astype(np.float)
	y = np.array(y).astype(np.float)
	print("Data Read !!!\n X : ", x.shape,"\t Y : ", y.shape)
	return (x,y)

def initialise_w(initialise):

	if(initialise == 'xavier'):
		if(num_class == 'multiclass'):
			w = np.random.randn(d,c)/np.sqrt(d*c)
		else:
			w = np.random.randn(d,1)/np.sqrt(d)
		print("w is initialised using xavier algorithm")

	elif(initialise == 'zeros'):
		if(num_class == 'multiclass'):
			w = np.zeros((d,c))
		else:
			w = np.zeros((d,1))		
		print("w is initialised as a zero vector")
	else:
		print(initialise, "Method unknown")

	return w 	#w.shape = (21,4)/(21,1)

def compute_mu(X, w):

	if(num_class == 'multiclass') :
		w = w.reshape(d,c)
	elif(num_class == 'oneclass') :
		w = w.reshape(d,1)
	mu = softmax(np.matmul(X,w), axis=1)
	#print("softmax",np.sum(np.sum(mu,axis=1)))
	return mu

def neg_log_posterior(w):

	mu = compute_mu(X, w)		# y.shape =  (872, 1) mu.shape =  (872, 1)
	log_like = np.sum(np.multiply(Y, np.log(mu+1e-5)))
	w_norm = np.sum(w**2)
	#w_norm = np.power(np.linalg.norm(w),2)
	neg_log_pos = -log_like+w_norm/2
	#print("neg_log_posterior = {:.4f} \tlog_like = {:.4f} \tw_norm = {:.4f}".format(neg_log_pos, log_like, w_norm))
	return(neg_log_pos)

def first_derivative(w):
	
	mu = compute_mu(X, w)	# mu.shape = (1000,4)
	return(np.matmul(np.transpose(X),(mu-Y)).flatten() + w)

def train_bfgs( X, Y, initialise = 'xavier'):

	w = initialise_w(initialise)
	res = minimize(neg_log_posterior, w, method='BFGS')#, jac=first_derivative)
	
	print("\n_____________Model trained______________\n")
	print("\nModel weights : ", res.x)
	if(num_class == 'multiclass'):
		correct,n = test(res.x, testX, testY)
		print("\n_____________Test Accuracy______________\n")		
		print("Correct : {0:4f} \t Accuracy : {1:4.2f} ".format(correct,float(correct)/n))
	return res.x

def test(w, X, y):

	n,d = X.shape
	print(w.shape)
	mu = compute_mu(X, w)

	if(num_class == 'multiclass'):
		true = np.argmax(y, axis=1)
		yhat = np.argmax(mu, axis=1)
		correct = np.sum(true == yhat)
	else :
		
		mu = mu.flatten()
		y = y.flatten()
		yhat = np.zeros((y.shape))
		yhat[mu>0.5] = 1
		correct = np.sum(y == yhat)	
	#correct = np.sum(y[yhat])
	return(correct,n)

def onevsall():

	global Y, num_class
	num_class = 'oneclass'
	w = np.array([])

	for i in range(c):

		print("Training {}th classifier".format(i))
		#Y = Y_copy[:,i].reshape(-1,1)
		w_i = train_bfgs(X, Y[:,i].reshape(-1,1))
		w = np.append(w,w_i)
		correct,n = test(w_i, testX, testY[:,i])
		correct,n = test(w_i, X, Y)
		print("\n_____________Test Accuracy for {}th classifier______________\n".format(i))		
		print("Correct : {0:4f} \t Accuracy : {1:4.2f} ".format(correct,float(correct)/n))

	w = w.reshape(d,4)
	#print(w.shape)
	print("\n One-vs-all accuracy \n")
	num_class = 'multiclass'
	correct,n = test(w, testX, testY)
	print("Model performance(onevsall) \nCorrect : {0:4d} \t Accuracy : {1:4.2f} ".format(correct,float(correct)/n))
	print("\n_____________Model trained______________\n")
	print("\nModel weights : ", w)


X, Y = read_csv(path+'train.csv')
X = X.astype(float)
testX, testY = read_csv(path+'test.csv')

np.random.seed(0)

n,d = X.shape
n,c = Y.shape
Y_copy = Y
num_class = args.num_class
initialise = args.initialise
#print(X[0:5], Y[0:5])
if(args.num_class == 'multiclass'):
	train_bfgs(initialise)
else:
	onevsall()