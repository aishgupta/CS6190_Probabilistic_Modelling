import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
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

	print("Data read !!!\n X : ", x.shape,"\t Y : ", y.shape)	# X.shape :  (872, 5) 	 y.shape :  (872,)
	return (x,y)

def norm_data(x):
	#To normalise the data within [-1, 1]
	x =  (x - np.mean(x, axis=0))*(1/(np.max(x,axis=0) - np.min(x, axis=0)))    #max(x) = 0.47, min(x) = -0.53
	return(x)
	
def get_data(filename):
	x, y = read_csv(filename)
	x[:, :-1] = norm_data(x[:, :-1])
	return(x,y)

def gaussian(x, m, c):
	return(multivariate_normal.pdf(x, mean=m, cov=c))

def compute_map(x, y, w0):
	def log_posterior(w):
		global x, y
		w = w.reshape((x.shape[1],1))
		y = y.reshape(y.shape[0],1)
		temp = expit(np.matmul(x, w))
		pos = np.sum(y*np.log(temp) + (1-y)*np.log(1-temp)) - np.matmul(w.transpose(), w)/2
		return(-pos)
	return(minimize(log_posterior, w0))

def get_precision(x, w_map):
	sigmoid = expit(np.matmul(x, w_map))
	prec = np.eye(x.shape[1])
	for i in range(x.shape[0]):
		temp1 = x[i].reshape(x[i].shape[0], 1)
		prec = prec + np.matmul(temp1, temp1.transpose()) * sigmoid[i] * (1 - sigmoid[i])
	return prec

#Laplace approximation
def laplace_approx(x,y):

	res = compute_map(x, y.reshape(y.shape[0], 1), np.zeros((x.shape[1], 1)))
	w_map = res.x
	prec = get_precision(x, w_map)
	cov = np.linalg.inv(prec)
	return w_map, cov
		
def compute_cov_meanfield(xi, x):
	lamb = compute_lambda(xi)
	lamb = lamb[:,np.newaxis]
	lamb = np.repeat(lamb, x.shape[1], axis=1)
	prec = 1/(2*(np.sum(np.multiply(np.multiply(x, x),lamb), axis = 0) + 0.5))
	s = np.multiply(np.eye(x.shape[1]), prec)
	return s

def compute_mean_meanfield(m, s, xi, x, y):

	y = np.repeat(y, x.shape[1], axis=1) - 0.5
	first_term = np.sum(np.multiply(x, y), axis=0)
	temp1 = np.multiply(x, np.repeat(m[np.newaxis,:], x.shape[0], axis=0))
	lamb = compute_lambda(xi)
	lamb = lamb[:, np.newaxis]
	lamb = np.repeat(lamb, x.shape[1], axis=1)
	xl = np.multiply(x, lamb)
	for i in range(x.shape[1]):
		temp2 = 0
		for j in range(x.shape[1]):
			if(j != i):
				temp2 += np.sum(np.multiply(temp1[:,j], xl[:,i]))
		second_term = -2*temp2
		m[i] = (first_term[i]+second_term)*s[i,i]
	return m

def variational_logistic_meanfield(x, y):
	max_iter = 100
	xi = -np.ones(x.shape[0])
	m = np.ones(x.shape[1])
	s = np.zeros((x.shape[1], x.shape[1]))
	#s = np.eye(x.shape[1])
	for i in range(max_iter):
		s = compute_cov_meanfield(xi, x)
		m = compute_mean_meanfield(m, s, xi, x, y)
		xi = compute_xi(x, s, m).reshape(x.shape[0])
	return m,s

def compute_mean(m0, s0, s, x, y):
	m0 = m0.reshape((m0.shape[0],1))
	temp1 = np.matmul(np.linalg.inv(s0),m0).reshape(m0.shape)
	temp2 = 0
	for i in range(x.shape[0]):
		temp2 = temp2+(y[i]-0.5)*x[i]
	temp2 = temp2.reshape(m0.shape)
	m = np.matmul(s, temp1 + temp2)
	return m

def compute_lambda(xi):
	temp = expit(xi)-0.5
	for i in range(xi.shape[0]):
		temp[i] = temp[i]/(2*xi[i] + 1e-5)
	return temp

def compute_xi(x, s, m):
	temp1 = s+np.matmul(m.reshape(-1,1), m.reshape(-1,1).transpose())
	xi = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		temp2 = np.matmul(x[i].reshape(1,-1), temp1)
		xi[i] = np.sqrt(np.matmul(temp2, x[i].reshape(1,-1).transpose()))
	return xi

def compute_cov(s0, xi, x):
	temp = 0
	lamb = compute_lambda(xi)
	for i in range(x.shape[0]):
		temp = temp+lamb[i]*np.matmul(x[i].reshape(-1,1), x[i].reshape(-1,1).transpose())
	s = np.linalg.inv(s0) + 2 * temp
	return np.linalg.inv(s)

#variational logistic inference
def variational_logistic(x, y):
	max_iter = 100    
	#x = np.ones((5,3))
	#y = np.array([1,0,0,1,1])
	xi = np.ones(x.shape[0])
	m0 = np.zeros(x.shape[1])
	s0 = np.eye(x.shape[1])
	for i in range(max_iter):
		s = compute_cov(s0, xi, x)
		m = compute_mean(m0, s0, s, x, y)
		xi = compute_xi(x, s, m).reshape(x.shape[0])
	return m,s

def gauss_hermite(mean, std):
	degree = 100
	points, weights = np.polynomial.hermite.hermgauss(degree)
	val = expit(points*np.sqrt(2)*std + mean)
	F = np.sum(val*weights)
	return F

def predictive_likelihood(x_test, y_test, m, s):
	likelihood = 0.0
	for i in range(x_test.shape[0]):
		x_i = x_test[i].reshape(x_test[i].shape[0],1)
		m = m.reshape(m.shape[0],1)
		mu = np.matmul(x_i.transpose(),m)
		std = np.sqrt(np.matmul(np.matmul(x_i.transpose(), s), x_i))
		prob_1 = gauss_hermite(mu, std)*(1/np.sqrt(np.pi))
		likelihood += prob_1 * y_test[i] + (1 - prob_1) * (1 - y_test[i])
	return (likelihood/x_test.shape[0])

def test(w, X, y):

	n,d = X.shape
	mu = expit(np.matmul(X,w))
	#print(mu.shape, n, d)
	yhat = np.zeros((n,1)).astype(np.float)
	yhat[mu>0.5]=1
	yhat = yhat.reshape(y.shape)
	correct = np.sum(yhat==y)
	return(correct,n)

if __name__ == '__main__':

	np.set_printoptions(precision=3)

	train_filename = "../data/bank-note/train.csv"
	test_filename = "../data/bank-note/test.csv"
	x, y = get_data(train_filename)
	x_test, y_test = get_data(test_filename)


	print("\n\n=====================================================================\n")
	print("Laplace approximation : ")
	w_map, cov = laplace_approx(x, y)
	(correct, n) = test(w_map, x_test, y_test)
	like = predictive_likelihood(x_test, y_test, w_map, cov)
	print("\nMean : \n", w_map)
	print("\nCovariance : \n", cov)
	print("\nTest Accuracy", correct / n)
	print("Predictive likelihood : ", like)

	#with hessian covariance
	print("\n\n=====================================================================\n")
	print("Laplace approximation with diagonal hessian : ")
	cov_hess = np.multiply(cov,np.eye(cov.shape[0]))
	like = predictive_likelihood(x_test, y_test, w_map, cov_hess)
	print("\nMean : \n", w_map)
	print("\nCovariance : \n", cov_hess)
	print("\nTest Accuracy", correct / n)
	print("Predictive likelihood : ", like)

	print("\n\n=====================================================================\n")
	print("Variational logistic regression inference using EM updates")
	w_map_var, cov_var = variational_logistic(x,y)
	w_map_var = w_map_var.squeeze()
	(correct, n) = test(w_map_var, x_test, y_test)
	like = predictive_likelihood(x_test, y_test, w_map_var, cov_var)
	print("\nMean : \n", w_map_var)
	print("\nCovariance : \n", cov_var)
	print("\nTest Accuracy", correct / n)
	print("Predictive likelihood : ", like)

	print("\n\n=====================================================================\n")
	print("Variational logistic regression meanfield inference using EM updates")
	w_map_varm, cov_varm = variational_logistic_meanfield(x, y)
	(correct, n) = test(w_map_varm, x_test, y_test)
	like = predictive_likelihood(x_test, y_test, w_map_varm, cov_varm)
	print("\nMean : \n", w_map_varm)
	print("\nCovariance : \n", cov_varm)
	print("\nTest Accuracy", correct / n)
	print("Predictive likelihood : ", like)



