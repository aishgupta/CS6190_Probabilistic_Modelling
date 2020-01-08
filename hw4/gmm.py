#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:38:26 2019

@author: aish
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

def read_data(filename):
    x = []
    f =  open(filename,'r')
    for line in f.readlines():
        x.append([float(i) for i in line.strip().split(" ")])
    x = np.array(x)     #x.shape = (272, 2)
    return(x)
    
def norm_data(x):
    #To normalise the data within [-1, 1]
    x =  (x - np.mean(x, axis=0))*(1/(np.max(x,axis=0) - np.min(x, axis=0)))    #max(x) = 0.47, min(x) = -0.53
    plt.scatter(x[:,0], x[:,1])
    plt.show()
    return(x)

def plot_data(x, cluster_id, k, mean, iter1):
    colours = 'rb'
    plt.clf()
    for i in range(k):
        temp = (cluster_id==i)
        plt.scatter(x[temp,0], x[temp,1])
    plt.plot(mean[:,0], mean[:,1],'bD')
    plt.savefig("/home/aish/Desktop/pml/homework4/iter_"+str(iter1)+".png")

def gmm_init(k):
    mean = np.array([[-1, -1],[1, 1]])
    covariance = np.tile(0.5 * np.eye(2), (k,1,1))
    mix = np.ones((k,1))/k
    print("Iniitialisation done. \n mean = ", mean, "\n covariance = ", covariance, "\n mixing coefficients = ", mix)
    return mean, covariance, mix

def gaussian(x, m, c):
    return(multivariate_normal.pdf(x, mean=m, cov=c))
    
def e_step(x, k, mean, covariance, mix):
    gamma = np.zeros((x.shape[0], k))
    #print(gamma.shape, np.sum(gamma))
    for i in range(k):
        gamma[:,i] = mix[i]*gaussian(x, mean[i], covariance[i])
    temp = np.tile(1/np.sum(gamma, axis=1), (2,1)).transpose()
    #print("gamma", np.sum(gamma*temp))
    return(gamma*temp)  
    
def m_step(x, k, gamma):
    mix = np.sum(gamma, axis=0)/np.sum(gamma)
    mean = np.zeros((k,x.shape[1]))
    covariance = np.zeros((k, x.shape[1], x.shape[1]))
    for i in range(k):
        #print(np.sum(gamma[:,i]))
        temp1 = gamma[:,i].reshape(gamma.shape[0],1)
        mean[i] = np.sum(x*temp1, axis=0)/np.sum(gamma[:,i])
        temp2 = x - mean[i]
#        temp2 = np.reshape(temp2, (temp2.shape[0], temp2.shape[1], 1))
#        temp3 = np.moveaxis(temp2, [0,1,2], [0,2,1])
        temp3 = 0
        for j in range(x.shape[0]):
            temp3 = temp3+gamma[j,i]*np.matmul(temp2[j].reshape(-1,1), temp2[j].reshape(-1,1).transpose())
#        covariance[i] = np.sum(np.matmul(temp2, temp3), axis=0) /np.sum(gamma[:,i])
        covariance[i] = temp3/np.sum(gamma[:,i])
    return mean, covariance, mix

def test_code():
    x = np.arange(10).reshape(5,2)
    k = 2
    mean, covariance, mix = gmm_init(k)
    #e_step(x,k,mean, covariance, mix)
    m_step(x, k, np.ones(x.shape)*0.2)
    
def gmm(k, filename):
    k = 2    
    max_iter = 100
    plot_iter = [1,2,5,100]
    x = read_data(filename)        
    x = norm_data(x)
    mean, covariance, mix = gmm_init(k)    
    for i in range(1, max_iter+1):
        gamma = e_step(x, k, mean, covariance, mix)
        mean, covariance, mix = m_step(x, k, gamma)
        #print(i, mean)
        if(i in plot_iter):
            cluster_id = np.argmax(gamma, axis=1)
            plot_data(x, cluster_id, k, mean, i)
#            print(mix)
#            print(covariance)
#            print(gamma)
    print(cluster_id.shape)
     
gmm(2, "data/faithful/faithful.txt")



    