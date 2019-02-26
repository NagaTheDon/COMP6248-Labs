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
    sum_term = (X.t()@(hypothesis(theta, X) - y)).sum(0)
    grad = (1/M)*sum_term
    return grad
    # raise NotImplementedError()
    
## cost_func computes
def cost_func(theta, X, y):
    # YOUR CODE HERE
    sum_term = ((hypothesis(theta, X) - y)**2).sum(0)
    cost = (1/(2*M))*sum_term
    return cost
    # raise NotImplementedError()

V_dw = 0
S_dw = 0
## The weight update computed using the ADAM optimisation algorithm
def weightupdate_adam(count, X, y):
	#lo refers to local variable in this function
    der_w = grad_cost_func(theta_0, X, y)
    V_dw_lo = beta_1*V_dw + (1 - beta_1)*der_w
    V_dw_corr = V_dw_lo/(1 - (beta_1**count))

    S_dw_lo = beta_2*S_dw + (1 - beta_2)*(der_w**2)
    S_dw_corr = S_dw_lo/(1 - (beta_2**count))
    theta = theta_0 - alpha*(V_dw_corr/torch.sqrt(torch.Tensor(S_dw_corr)))

    # if(count%10 == 0):
    # 	print("Cost function at count ", count, " is " ,cost_func(theta, X,y))
    return V_dw_lo, S_dw_lo, theta


## The weight update computed using SGD + momentum

def weightupdate_sgd_momentum(count, X, y):
	#lo refers to local 
	der_w = grad_cost_func(theta_0, X, y)
	V_t_lo = (beta_1*V_t) + (1 - beta_1)*der_w

	theta = theta_1 - (alpha*V_t_lo) # This is basically w
	# if(count%10 == 0):
	# 	print("Cost function at count ", count, " is " ,cost_func(theta, X,y))
	return V_t_lo, theta


## The weight updated computed using SGD
def weigthupdate_sgd(count, X, y):
	der_w = grad_cost_func(theta_0, X, y)
	theta = theta_2 - (alpha*der_w)
	# if(count%10 == 0):
	# 	print("Cost function at count ", count, " is " ,cost_func(theta, X,y))
	return theta


N = 200
beta_1 = 0.9
beta_2 = 0.999
alpha = 0.01

# Need to ask people in lab: for 200 iterations -> N+1 isn't it?
#Adam
theta_0 = torch.Tensor([[2],[4]]) #initialise
for i in range(1,N):
    V_dw, S_dw, theta_0 = weightupdate_adam(i, X, y)

#weightupdate_sgd_momentum
theta_1 = torch.Tensor([[2],[4]])
V_t = 0
for i in range(1,N):
	V_t, theta_1 =  weightupdate_sgd_momentum(i, X, y)

#weightupdate_sgd
theta_2 = torch.Tensor([[2],[4]])
for i in range(1,N):
	theta_2 =  weigthupdate_sgd(i, X, y)

print(theta_0, theta_1, theta_2)

theta_0_vals = np.linspace(-2,4,100)
theta_1_vals = np.linspace(0,4,100)
theta = torch.Tensor(len(theta_0_vals),2)

# Compute the value of the cost function, J, over all the thetas in order to plot the contour below.
J = torch.zeros(100,100)

for zero_ind in range(100):
    for first_ind in range(100):
        single_theta = torch.Tensor([[theta_0_vals[zero_ind]],
                               [theta_1_vals[first_ind]]])
        J[first_ind,zero_ind] = cost_func(single_theta, X, y)# Might need to change this... 

xc,yc = np.meshgrid(theta_0_vals, theta_1_vals)
contours = plt.contour(xc, yc, J, 20)
adam_point = plt.scatter(theta_0[0], theta_0[1], marker='o', s=20, c='red')
sgd_m_point = plt.scatter(theta_1[0], theta_1[1], marker='o', s=20, c='cyan')
sgd = plt.scatter(theta_2[0], theta_2[1], marker='o', s=20, c='green')

plt.legend((adam_point, sgd_m_point, sgd),
           ('ADAM', 'SGD with Momentum', 'SGD'),loc='lower right',
            fontsize=8)

plt.show()


# Now plot the output of SGD, momentum and Adam all on the same plot for comparison
