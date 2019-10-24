# -*- coding: utf-8 -*-
# Code for Gaussian process (GP) with propensity score (PS) correction from the
# paper "Debiased Bayesian inference for average treatment effects" at NeurIPS 2019
# The code is for the synthetic dataset - the code for the semi-synthetic dataset is commented out
import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from sklearn.linear_model import LogisticRegression
import time
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing

# Method parameters
cor_size = 0.2  # ratio of SD of the prior correction term to the GP kernel size (see Section 4)
jitter = 1e-9  # min eigenvalue enforced for GP covariance matrix
PS_bound = 0.1  # propensity score is bounded away from 0,1
num_sims = 20  # number of simulations
num_post = 2000  # number of posterior GP samples
cred = 0.95 # posterior probability of credible interval

ATE_data = np.zeros((num_sims,4))  # key quantitites about ATE posterior
ATE_data_noRand = np.zeros((num_sims,4))  # same without randomizing the feature distribution F (plug-in empirical distribution)

##################### Synthetic dataset #######################

# Simulation parameters
d=100  # number of features
n=500  # sample size        
sig_n = 1.0  # noise SD
ATE_true = 1.0  #true value of ATE (computed analytically)

# true propensity score
def prop_score(x):
    if isinstance(x,(list,np.ndarray)):
        if len(x)>=5:
            val = x[0]+(x[1]-0.5)**2 + x[2]**2 - 2*np.sin(2*x[3]) + np.exp(-x[4])-np.exp(-1.0)+1.0/6.0
            if val > 0:
                return 1.0
            else:
                return 0.0
    
# true regression function
def mreg(x,r):
    if isinstance(x,(list,np.ndarray)):
        if len(x)>=5:
            val = np.exp(-x[0])+x[1]**2+x[2]+np.cos(x[4]) + r*(1+2*x[1]*x[4])  #heterogeneous
#            val = np.exp(-x[0])+x[1]**2+x[2]+np.cos(x[4]) + r  #homogeneous
            if x[3]>0:  #indicator function 1{x[3]>0}
                val = val+1.0
            return val

# Treatment effect function (= difference when R=0,1)
def Lmreg(x):
    return mreg(x,1)-mreg(x,0)


##################### Semi-synthetic dataset #######################

##Import causal inference data
#obs = pd.read_csv('ihdp.csv')
#cov_cts = ["bw","b.head","preterm","birth.o","nnhealth","momage"]
#cov_bin = ["sex","twin","b.marr","mom.lths","mom.hs",	"mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was"]
#
#X_cts = obs.loc[:,cov_cts].values
#X_bin = obs.loc[:,cov_bin].values
#X = obs.loc[:,[*cov_cts,*cov_bin]].values # ordered as [*cov_cts,*cov_bin]
#R = obs.loc[:,'treat'].values[:,None]
#
#X = preprocessing.scale(X) # Standardize covariates
#Z = np.column_stack((X,R)) # combine the inputs as a single regression matrix
#
## Parameters
#d=X.shape[1] #dimension of inputs
#n=X.shape[0] # sample size
#sig_n = 1.0 #noise sd
#ATE_true = 4.0  #true value of ATE (computed analytically)

##################### GP with propensity score correction #######################

start = time.time()

for run in range(num_sims):
    
    ### Synthetic data ###
    # Generate data - X=features, R=treatment assignment, Y=outcome, Z=(X,R)
    X = np.random.normal(0,1,(n,d))
    R = np.asarray([np.random.binomial(1,prop_score(X[i,:]),1) for i in range(n)])
    Y = np.asarray([mreg(X[i,:],R[i]) + sig_n*np.random.normal(0,1) for i in range(n)])
    Z = np.column_stack((X,R))  # combine features with treatment assignments to get 'design' matrix
    
#    ### Semi-synthetic data ###
#    theta = np.random.choice([0.0,0.1,0.2,0.3,0.4],size=d+1,replace=True,p=[0.6,0.1,0.1,0.1,0.1])
#    yb0hat = np.exp(np.matmul(np.column_stack((np.ones(n),X+0.5*np.ones((n,d)))), theta))
#    yb1hat = np.matmul(np.column_stack((np.ones(n),X)), theta)
#    offset = np.mean(yb1hat - yb0hat) -4.0
#    yb1hat = yb1hat - offset*np.ones(n)
#
#    Y0 = np.random.normal(yb0hat,sig_n*np.ones(n))
#    Y1 = np.random.normal(yb1hat,sig_n*np.ones(n))
#
#    Y = np.zeros(n)[:,None]
#    for i in range(n):
#        if R[i]==0:
#            Y[i] = Y0[i]
#        elif R[i]==1:
#            Y[i] = Y1[i]   
    
    # Estimate the propensity score using (truncated) logistic regression
    LR = LogisticRegression(solver='lbfgs').fit(X,np.ravel(R))
    prop = LR.predict_proba(X)[:,1]
    prop = np.asarray([max(min(prop[i],1.0-PS_bound),PS_bound) for i in range(n)]) #truncation
    
    # Fit a squared-exponential GP without PS correction to estimate the hyperparameters
    k = GPy.kern.RBF(d+1,active_dims=list(range(d+1)),name='rbf',ARD=True)  #GP kernel
    m = GPy.models.GPRegression(Z,Y,k)
    m.optimize()
    
    # Since we use the Bayesian bootstrap (BB), need only evaluate the posterior GP at data points plus their counterfactuals
    Z_BB = np.hstack((np.repeat(X,2,axis=0),np.asarray([0.0,1.0]*n).reshape(2*n,1)))
    data_filter = [False]*(2*n)  # True when Z_BB equals factual, false when equals counterfactual
    for i in range(n):
        if R[i]==0:
            data_filter[2*i] = True
        elif R[i]==1:
            data_filter[2*i+1] = True
    
    PS_weight = np.zeros(2*n)[:,None]
    for i in range(n):
        PS_weight[2*i] = -1.0/(1.0-prop[i])  # R=0
        PS_weight[2*i+1] = 1.0/prop[i]  # R=1
    
    M = np.mean(abs(PS_weight[data_filter]))  # mean PS reweighting at the observations
    cor_var = (cor_size**2)*m.rbf.variance/((M**2)*n)  # variance of correction term (\nu_n^2 in paper)

    # Prior covariance matrix with PS correction - make sure min eigenvalue >= jitter for numerical stability
    PriorCov_BB = k.K(Z_BB,Z_BB) + cor_var*np.matmul(PS_weight,PS_weight.T)
    PriorCov_eig = min(np.linalg.eigvals(PriorCov_BB))
    if PriorCov_eig < jitter:
        PriorCov_BB = PriorCov_BB +(jitter - PriorCov_eig)*np.eye(2*n)

    # Cholesky factorization of K(Z,Z)+sig_n^2 I
    PriorCholesky = np.linalg.cholesky(PriorCov_BB[data_filter,:][:,data_filter]+m.Gaussian_noise.variance*np.eye(n))
    alpha = solve_triangular(PriorCholesky.T,solve_triangular(PriorCholesky,Y,lower=True),lower=False) #Solves (K(Z,Z)+sig_n^2 I)\alpha=Y 
    beta = np.zeros((n,2*n))
    for i in range(2*n):
        beta[:,i]=solve_triangular(PriorCholesky,PriorCov_BB[data_filter,:][:,i],lower=True)

    meanGP = np.matmul(PriorCov_BB[:,data_filter],alpha)  # posterior mean
    CovGP = PriorCov_BB - np.matmul(beta.T,beta)  # posterior covariance
    
    # Compute covariance matrix of Lmreg(x) at data points (only interested in difference - reduces size from 2n to n)
    meanLm = np.zeros((n,1))
    CovLm = np.zeros((n,n))
    for i in range(n):
        meanLm[i] = meanGP[2*i+1]-meanGP[2*i]
        for j in range(i+1):
            CovLm[i,j] = CovGP[2*i+1,2*j+1] -CovGP[2*i+1,2*j] - CovGP[2*i,2*j+1] + CovGP[2*i,2*j]
            CovLm[j,i] = CovLm[i,j]

    # Ensure min eigenvalue >= jitter for numerical stability
    min_eig = min(np.linalg.eigvals(CovLm))
    if min_eig < jitter:
        CovLm = CovLm + (jitter-min_eig)*np.eye(n)

    # Posterior draws for the ATE: do a Cholesky decomposition once and draw iid standard normals
    Chol_Lm = np.linalg.cholesky(CovLm)
    ATE = np.zeros(num_post)  # ATE posterior draws
    ATE_noRand = np.zeros(num_post)  # ATE posterior draws without randomizing the feature distribution F
    for i in range(num_post):
        DP_weights = np.random.exponential(1,n)
        DP_weights = DP_weights/sum(DP_weights)  #draw from Dir(1,...,1) distribution
        GP_draw = meanLm + np.matmul(Chol_Lm,np.random.normal(0,1,(n,1)))
        ATE[i]=np.dot(DP_weights,GP_draw)  # using the Bayesian bootstrap
        ATE_noRand[i] = np.dot(np.ones(n),GP_draw)/n  # using the empirical distribution (no randomization of F)
        
    # Store ATE posterior quantities results from this run
    ATE_data[run,0]=np.mean(meanLm)  # posterior mean
    low =np.quantile(ATE,(1-cred)/2)  #lower point of credible interval
    up =np.quantile(ATE,(1+cred)/2)  #upper point of credible interval
    ATE_data[run,1] = up-low  # width of credible interval
    if ATE_true >= low and ATE_true <= up:
        ATE_data[run,2] = 1.0  # 1 if true ATE falls within credible interval, 0 otherwise
    if low <= 0 and up>=0:
        ATE_data[run,3] = 1.0  # if 0 is contained in cred. interval (Type II error)
    
    # Same for method without randomizing F
    ATE_data_noRand[run,0]=np.mean(meanLm)  # posterior mean
    low_noRand = np.quantile(ATE_noRand,(1-cred)/2)  #lower point of credible interval
    up_noRand = np.quantile(ATE_noRand,(1+cred)/2)  #upper point of credible interval
    ATE_data_noRand[run,1] = up_noRand-low_noRand  # width of credible interval
    if ATE_true >= low_noRand and ATE_true <= up_noRand:
        ATE_data_noRand[run,2] = 1.0  # 1 if true ATE falls within credible interval, 0 otherwise
    if low_noRand <= 0 and up_noRand>=0:
        ATE_data_noRand[run,3] = 1.0  # if 0 is contained in cred. interval (Type II error)
    
    print(run,time.time()-start)

#    # Plotting posterior draws if desired
#    plt.figure()
#    plt.hist(ATE,bins='auto',density=True)  #histogram of posterior draws
#    plt.axvline(x=np.sum(ATE)/num_post,color='black',lw=2)  #posterior mean
#    plt.axvline(x=np.quantile(ATE,(1-cred)/2),color='black',lw=2.0,ls='--')  #CI
#    plt.axvline(x=np.quantile(ATE,(1+cred)/2),color='black',lw=2.0,ls='--')  #CI
#    plt.axvline(x=ATE_true,color='red',lw=2.0)  #true ATE
#    lim = plt.xlim()
#    pts = np.linspace(lim[0],lim[1],100)
#    plt.plot(pts,stats.norm.pdf(pts,np.sum(ATE)/num_post,np.std(ATE)),color='darkorange',lw=2.0)
#    plt.savefig('GP_PS.png',dpi=900)
    
# Print summary statistics to console
print("\n")
print("With randomization of F")
print("Average absolute error of posterior mean: {} plus/minus {}".format(np.mean(abs(ATE_data[:,0]-ATE_true)),np.std(abs(ATE_data[:,0]-ATE_true))))
print("Average CI size: {} plus/min {}".format(np.mean(ATE_data[:,1]),np.std(ATE_data[:,1])))
print("Average coverage: {} Average Type II error: {}".format(np.mean(ATE_data[:,2]),np.mean(ATE_data[:,3])))

print("\n")
print("Without randomization of F")
print("Average absolute error of posterior mean: {} plus/minus {}".format(np.mean(abs(ATE_data_noRand[:,0]-ATE_true)),np.std(abs(ATE_data_noRand[:,0]-ATE_true))))
print("Average CI size: {} plus/min {}".format(np.mean(ATE_data_noRand[:,1]),np.std(ATE_data_noRand[:,1])))
print("Average coverage: {} Average Type II error: {}".format(np.mean(ATE_data_noRand[:,2]),np.mean(ATE_data_noRand[:,3])))

## Save full information to CSV files
#np.savetxt("GP with PS n={}.csv".format(n), ATE_data, delimiter=",")
#np.savetxt("GP with PS noRand n={}.csv".format(n), ATE_data_noRand, delimiter=",")