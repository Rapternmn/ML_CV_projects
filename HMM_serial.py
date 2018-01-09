# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 10:18:13 2016

@author: Elena Sharova
"""
import numpy as np
import pandas as pd
import json as js
from yahoo_finance import Share
import datetime as dt
import pandas as pd
import pandas_datareader.data as web


def test_model(a,b,seq):

    N = np.shape(a)[0]
    T = np.shape(seq)[0]

    alpha = np.zeros((N,T))
    alpha[:,0] = np.dot(a,b[:,seq[0]])


    for t in xrange(1,T):
        for i in xrange(N):
            alpha[i,t] = b[i,seq[t]] * np.sum(alpha[:,t-1] * a[:,i])

    print alpha


class HMM(object):
    # Implements discrete 1-st order Hidden Markov Model 

    def __init__(self, tolerance = 1e-6, max_iterations=10000, scaling=True):
        self.tolerance=tolerance
        self.max_iter = max_iterations
        self.scaling = scaling

    def HMMfwd(self, a, b, o, pi):
        # Implements HMM Forward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]

        # print "T : {}".format(T)
    
        alpha = np.zeros((N,T))
        # initialise first column with observation values
        alpha[:,0] = pi*b[:,o[0]]
        c = np.ones((T))
        
        if self.scaling:
            
            c[0]=1.0/np.sum(alpha[:,0])
            alpha[:,0]=alpha[:,0]*c[0]
            
            for t in xrange(1,T):
                c[t]=0
                for i in xrange(N):
                    alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
                c[t]=1.0/np.sum(alpha[:,t])
                alpha[:,t]=alpha[:,t]*c[t]

        else:
            for t in xrange(1,T):
                for i in xrange(N):
                    alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
        
        return alpha, c

    def HMMbwd(self, a, b, o, c):
        # Implements HMM Backward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        beta = np.zeros((N,T))
        # initialise last row with scaling c
        beta[:,T-1] = c[T-1]
    
        for t in xrange(T-2,-1,-1):
            for i in xrange(N):
                beta[i,t] = np.sum(b[:,o[t+1]] * beta[:,t+1] * a[i,:])
            # scale beta by the same value as a
            beta[:,t]=beta[:,t]*c[t]

        return beta

    def HMMViterbi(self, a, b, o, pi):
        # Implements HMM Viterbi algorithm        
        
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        path = np.zeros(T)
        delta = np.zeros((N,T))
        phi = np.zeros((N,T))
    
        delta[:,0] = pi * b[:,o[0]]
        phi[:,0] = 0
    
        for t in xrange(1,T):
            for i in xrange(N):
                delta[i,t] = np.max(delta[:,t-1]*a[:,i])*b[i,o[t]]
                phi[i,t] = np.argmax(delta[:,t-1]*a[:,i])
    
        path[T-1] = np.argmax(delta[:,T-1])
        for t in xrange(T-2,-1,-1):
            path[t] = phi[int(path[t+1]),t+1]
    
        return path,delta, phi

 
    def HMMBaumWelch(self, o, N,a,b, dirichlet=False, verbose=False, rand_seed=1):
        # Implements HMM Baum-Welch algorithm        
        
        T = np.shape(o)[0]

        M = int(max(o))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

        digamma = np.zeros((N,N,T))

    
        # Initialise A, B and pi randomly, but so that they sum to one
        np.random.seed(rand_seed)
        
        # Initialisation can be done either using dirichlet distribution (all randoms sum to one) 
        # or using approximates uniforms from matrix sizes
        if dirichlet:
            pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))
            
            a = np.random.dirichlet(np.ones(N),size=N)
            
            b=np.random.dirichlet(np.ones(M),size=N)
        else:
            
            pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
            pi=1.0/N*np.ones(N)-pi_randomizer

        
        error = self.tolerance+10
        itter = 0
        while ((error > self.tolerance) & (itter < self.max_iter)):   

            prev_a = a.copy()
            prev_b = b.copy()
    
            # Estimate model parameters
            alpha, c = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o, c) 
    
            for t in xrange(T-1):
                for i in xrange(N):
                    for j in xrange(N):
                        digamma[i,j,t] = alpha[i,t]*a[i,j]*b[j,o[t+1]]*beta[j,t+1]
                digamma[:,:,t] /= np.sum(digamma[:,:,t])
    

            for i in xrange(N):
                for j in xrange(N):
                    digamma[i,j,T-1] = alpha[i,T-1]*a[i,j]
            digamma[:,:,T-1] /= np.sum(digamma[:,:,T-1])
    
            # Maximize parameter expectation
            for i in xrange(N):
                pi[i] = np.sum(digamma[i,:,0])
                for j in xrange(N):
                    a[i,j] = np.sum(digamma[i,j,:T-1])/np.sum(digamma[i,:,:T-1])
    	

                for k in xrange(M):
                    filter_vals = (o==k).nonzero()
                    b[i,k] = np.sum(digamma[i,:,filter_vals])/np.sum(digamma[i,:,:])
    
            error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
            itter += 1            
            
            if verbose:            
                print ("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
        return a, b, pi, alpha

if __name__ == '__main__':

    hmm = HMM()
    
    hist = []

    # fname = 'blink_new.txt'

    fname = 'no_blink.txt'

    if fname == 'no_blink.txt':

        print 'IN No Blink'

        with open(fname,'r') as f:

            lines = f.readlines()

            i = 0

            for line in lines:
                line = int(line.strip())
                hist.append(line)
    else :

        with open(fname,'r') as f:

            lines = f.readlines()
            no_lines = len(lines)
            i = 0

            while(i<no_lines):
                line = int(lines[i].strip())

                if (line == 0):

                    arr_loc = []

                    if(i>6) :
                        arr_loc = [ int(line.strip()) for line in lines[i-6:i+7] ]
                        i+=6
                    else:
                        arr_loc = [ int(line.strip()) for line in lines[:13] ]
                        i+=13

                    hist.append(arr_loc)
                else:
                    i+=1

    hist = np.array(hist)

    if fname == 'no_blink.txt':
        seq = hist[200:213]

    else:
        seq = hist[0]

    hist =  hist.flatten()

    print hist

    N = 3
    M = 3

    a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
    a=1.0/N*np.ones([N,N])-a_randomizer

    b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
    b = 1.0/M*np.ones([N,M])-b_randomizer

    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist, N,a,b, False, True)

    print a
    print b
    # (path, delta, phi)=hmm.HMMViterbi(a, b, hist, pi_est)

    # print seq
    # print type(seq)
    # print seq.shape

    # blink_a = np.array([[ 0.84374422, 0.15625578],[ 0.92962214, 0.07037786]])
    # blink_b = np.array([[ 3.16330020e-01, 6.83669980e-01, 5.18945352e-12],[  3.44655471e-04, 9.39448362e-12, 9.99655345e-01]])
    

    # # no_blink_a = np.array([[ 0.07056403,  0.92943597],[ 0.10376217,  0.89623783]])
    # # no_blink_b = np.array([[ 1.23906669e-51,   8.28520530e-04,   9.99171479e-01],[2.35273694e-01 ,  7.64726306e-01 ,  1.53157979e-15]])

    # no_blink_a = np.array([[ 0.97714286 ,0.02285714],[ 0.10376217,  0.89623783]])
    # no_blink_b = np.array([[  9.99999996e-01  , 3.80703558e-09 ,  7.94072006e-12],[  2.66319207e-12  , 7.15210355e-01 ,  2.84789645e-01]])

    # test_model(blink_a,blink_b,seq)
    # test_model(no_blink_a,no_blink_b,seq)



