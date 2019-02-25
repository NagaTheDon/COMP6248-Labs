import torch
import numpy as np
import matplotlib.pyplot as plt
import math

## generate M data points roughly forming a line (noise added)
M = 50
theta_true = torch.Tensor([[0.5], [2]])
# print (theta_true)

X = 10 * torch.rand(M, 2) - 5
X[:, 1] = 1.0

y = torch.mm(X, theta_true) + 0.3 * torch.randn(M, 1)

## hypothesis computes $h_theta$
def hypothesis(theta, X):
    # YOUR CODE HERE
    h_theta = torch.mm(X, theta)
    return h_theta
    # raise NotImplementedError()

## grad_cost_func computes the gradient of J for linear regression given J is the MSE 
def grad_cost_func(theta, X, y): 

    hyp = hypothesis(theta, X)
    sum_term = X.t()@(hyp- y)
    grad = (1/M)*sum_term
    return grad
    # raise NotImplementedError()
    
## cost_func computes
def cost_func(theta, X, y):
    # YOUR CODE HERE
    sum_term = (hypothesis(theta, X) - y)**2
    cost = (1/(2*M))*sum_term
    return cost
    # raise NotImplementedError()

## The weight update computed using the ADAM optimisation algorithm
def weightupdate_adam(count, X, y):

    der_w = grad_cost_func(theta_0, X, y)
    V_dw_lo = beta_1*V_dw + (1 - beta_1)*der_w
    V_dw_corr = V_dw_lo/(1 - (beta_1**count))

    S_dw_lo = beta_2*S_dw + (1 - beta_2)*(der_w**2)
    S_dw_corr = S_dw_lo/(1 - (beta_2**count))
    theta = theta_0 - alpha*(V_dw_corr/torch.sqrt(torch.Tensor(S_dw_corr)))

    print("Cost function at count ", count, " is " ,torch.sum(cost_func(theta, X,y)))
    return V_dw_lo, S_dw_lo, theta


## The weight update computed using SGD + momentum
# def weightupdate_sgd_momentum(count, X, y):


## The weight updated computed using SGD
# def weigthupdate_sgd(count, X, y):

N = 200
beta_1 = 0.9
beta_2 = 0.999
alpha = 0.01

theta_0 = torch.Tensor([[2],[4]]) #initialise

# Write the code that will call of the optimisation update functions and compute weight updates for each individual data point over N iterations.

V_dw = 0
S_dw = 0

for i in range(1,N):
    V_dw, S_dw, theta_0 = weightupdate_adam(i, X, y)

# raise NotImplementedError()

theta_0_vals = np.linspace(-2,4,100)
theta_1_vals = np.linspace(0,4,100)
theta = torch.Tensor(len(theta_0_vals),2)

# Compute the value of the cost function, J, over all the thetas in order to plot the contour below.
# YOUR CODE HERE

# raise NotImplementedError()

# xc,yc = np.meshgrid(theta_0_vals, theta_1_vals)
# contours = plt.contour(xc, yc, J, 20)

# Now plot the output of SGD, momentum and Adam all on the same plot for comparison
# YOUR CODE HERE

# raise NotImplementedError()