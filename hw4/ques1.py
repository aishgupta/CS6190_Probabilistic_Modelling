'''
This file contains sample code about how to use Gauss–Hermite quadrature to compute a specific type of integral numerically.

The general form of this type of integral is:( see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature for more details)

F = int_{ -inf}^{+inf} e^{-x*x) f(x) dx,  (1)

in which we're calculating the integral of f(x) in the range ( -inf, +inf) weighted by e^(-x*x ).
Note that for f(x) being polynomial function, this integral is guaranteed to converge. But for some others convergence is not guaranteed.
'''

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import csv

def gass_hermite_quad( f, degree):
	'''
	Calculate the integral (1) numerically.
	:param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
	:param degree: integer, >=1, number of points
	:return:
	'''
	points, weights = np.polynomial.hermite.hermgauss( degree)	# points.shape = weights.shape = (degree,)
	#function values at given points
	f_x = f( points)	# f_x.shape = (degree,)
	#weighted sum of function values
	F = np.sum( f_x  * weights)
	return F

def ques1(x):
	# returns exp(-x^2)*sigmoid(10x+3)
	return(np.exp(-x*x)*expit(10*x + 3))

def ques1_only_sigmoid(x):
	# returns sigmoid(10x + 3) for function exp(-x^2)*sigmoid(10x+3)
	return(expit(10*x + 3))

def get_lambda(xi):
	lamb = -(1/(2*(xi*10+3)))*(expit(10*xi+3) - 0.5)
	return lamb

def gaussian(mean, std, x):
	return(norm.pdf(x, loc = mean, scale=std))

def compute_gauss_hermite_approx(x):

	F = gass_hermite_quad(ques1_only_sigmoid, degree=100)
	print("For degree = 100, the normalisation constant of the function is : ", F)
	return (ques1(x)/F)

def plot_density_curve(func, x_min, x_max):

	x = np.linspace(x_min, x_max, 500)
	y = func(x)
	plt.plot(x,y)

def get_MAP():

	def neg_log_ques1(x):
		y = -x*x + np.log(expit(10*x+3))
		return(-y)

	res = minimize(neg_log_ques1, np.array(0))
	map = res.x
	return(map)

def compute_laplace_approx(x):

		mean = get_MAP()
		sigmoid = expit(10*mean+3)
		var =  1/(2 + 100 * sigmoid*(1-sigmoid))
		y = gaussian(mean, math.sqrt(var), x)
		print(("Mean : {}, variance : {}").format(mean, var))
		return(y)

def compute_var_local_inference(x):

	xi = 0
	degree = 100
	Z1 = gass_hermite_quad(ques1_only_sigmoid, degree)
	def get_sigmod_y(x):

		lamb = get_lambda(xi)
		sigmoid_y = expit(10*xi+3) * np.exp(5 * (x - xi) + lamb * np.multiply(10*(x-xi), 10*(x+xi)+6))
		return sigmoid_y

	for i in range(100):
		qx = get_sigmod_y(x)*np.exp(-x*x)/Z1
		xi = x[np.argmax(qx)]
	return qx

if __name__ == '__main__':

	print("=====================================================================\n")
	print("Gauss-Hermite Quadrature approximation : ")
	plot_density_curve(compute_gauss_hermite_approx, -5 ,5)
	print("\n=====================================================================")
	print("\n\n=====================================================================\n")
	print("Laplace approximation : ")
	plot_density_curve(compute_laplace_approx, -5 ,5)
	print("\n=====================================================================")
	print("\n\n=====================================================================\n")
	print("Variational local inference : ")
	plot_density_curve(compute_var_local_inference, -5, 5)
	plt.legend(["Gauss Hermite", "Laplace Approx", "Local Variational"])
	plt.savefig("ques1.png")




