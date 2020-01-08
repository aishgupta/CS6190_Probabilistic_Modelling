import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import os

path = os.getcwd()
path = path+"/images"
try:
    os.mkdir(path)
except:
    print (path, " exists")
else:
    print ("Successfully created the directory ", path)

def gibbs(mu, covariance, n_iter, z):
    n = mu.shape[0]
    accepted = np.array(z)[np.newaxis,:]
    for i in range(n_iter):
        for j in range(n):
            k = (j+1)%2
            mu_j = mu[j] + covariance[j][k]*(z[k] - mu[k])/covariance[k][k]
            var_j = covariance[j][j] - covariance[j][k]*covariance[k][j]/covariance[k][k]
            z[j] = norm(loc = mu_j, scale=np.sqrt(var_j)).rvs(1)
            accepted = np.vstack((accepted, z))
    return accepted

def dU_dz(mu, cov, z):
    z = np.array(z-mu)
    grad = np.matmul(np.linalg.inv(cov),z)
    return grad

def leapfrog(z, r, s, mu, cov, eps, L):

    for i in range(L):
        r -= (eps/2)*dU_dz(mu, cov, np.copy(z))
        z += eps*np.matmul(np.linalg.inv(s), r)
        r -= (eps/2)*dU_dz(mu, cov, np.copy(z))
    return (z, r)

def accept_prob(pos_dist, current_state, next_state, mu, cov, s):
    current_state_p = pos_dist(current_state, mu, cov, s)
    next_state_p = pos_dist(next_state, mu, cov, s)
    return(np.min([1, next_state_p/current_state_p]))

def total_energy(state, mu, cov, s):
    z = state[0]
    r = np.array(state[1])
    z = np.array(z-mu)
    u = 0.5*(np.matmul(np.matmul(z.transpose(),np.linalg.inv(cov)), z))
    k = 0.5*(np.matmul(np.matmul(r.transpose(),np.linalg.inv(s)), r))
    return(np.exp(-u-k))

def hybrid_monte_carlo(mu, cov, burn_in, n_iter, eps, L, z):
    s = np.eye(2)
    r = multivariate_normal(mean=np.zeros(2), cov=s)
    mu = mu[:,np.newaxis]
    z_p = z[:,np.newaxis]
    rejected = np.array(z_p)
    accepted = np.array(z_p)
    for i in range(1, burn_in + 1):
        r_p = r.rvs(1)[:, np.newaxis]  # sampling r from normal distribution
        z_n, r_n = leapfrog(np.copy(z_p), np.copy(r_p), s, mu, cov, eps, L)
        r_n *= (-1)
        prob = accept_prob(total_energy, [z_p, r_p], [z_n, r_n], mu, cov, s)
        u = np.random.uniform(0, 1, 1)
        if (u <= prob):
            z_p = z_n
    print("Burn-in for " + str(burn_in) + " iterations done!")

    for i in range(1, n_iter + 1):
        accept = False
        r_p = r.rvs(1)[:, np.newaxis]  # sampling r from normal distribution
        z_n, r_n = leapfrog(np.copy(z_p), np.copy(r_p), s, mu, cov, eps, L)
        r_n *= (-1)
        prob = accept_prob(total_energy, [z_p, r_p, s], [z_n, r_n, s], mu, cov, s)
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
    return accepted.transpose() #, rejected.transpose()

def get_samples(func_name, mu, covariance, burn_in, n_iter, eps, L, z_init):
    if(func_name == gibbs):
        accepted = func_name(mu, covariance, n_iter//m, z_init)
        filename="ques2_gibbs.png"
    else:
        accepted = func_name(mu, covariance, burn_in, n_iter, eps, L, z_init)
        filename = "ques2_hmc.png"
    true_dist = multivariate_normal(mean=mu, cov=covariance).rvs(500)
    plt.clf()
    plt.scatter(true_dist[:, 0], true_dist[:, 1])
    plt.xlabel("z0")
    plt.ylabel("z1")
    plt.grid()
    plt.savefig(path+"/ques2_truedist.png")
    plt.plot(accepted[:, 0], accepted[:, 1], 'ro')
    plt.savefig(path+"/"+filename)

n = 2
mu = np.zeros(n)
z_init = np.array([-4, -4], dtype=np.float)
n_iter = 1000
eps = 0.1; L = 20; m = 10
burn_in = 100000
covariance = np.array([[3, 2.9], [2.9, 3]])
get_samples(gibbs, mu, covariance, burn_in, n_iter, eps, L, z_init)
get_samples(hybrid_monte_carlo, mu, covariance, burn_in, n_iter, eps, L, z_init)



