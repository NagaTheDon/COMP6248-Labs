import torch
import matplotlib.pyplot as plt
import numpy as np

## generate M data points roughly forming a line (noise added)
M = 100
theta_true = torch.Tensor([[0.5], [2]])

X = 10 * torch.rand(M, 2) - 5
X[:, 1] = 1.0

y = torch.mm(X, theta_true) + 0.3 * torch.randn(M, 1)
# print (y)

## visualise the data by plotting it
# YOUR CODE HERE
print (theta_true.shape)
# print (X)
print (X.shape)
print (y.shape)
plt.scatter(X[:,0].numpy(), y.numpy())
plt.show()


## hypothesis computes $h_theta$
def hypothesis(theta, X):
    # YOUR CODE HERE
    h_theta = torch.mm(X, theta)
    return h_theta
    # raise NotImplementedError()

## grad_cost_func computes the gradient of J for linear regression given J is the MSE 
def grad_cost_func(theta, X, y): 
    # YOUR CODE HERE
    X_t = X.t()
    sum_term = X.t()@(hypothesis(theta, X) - y)
    grad = (1/M)*sum_term
    return grad
    # raise NotImplementedError()

## cost_func computes the cost function J
def cost_func(theta, X, y):
    # YOUR CODE HERE
    sum_term = (hypothesis(theta, X) - y)**2
    cost = (1/(2*M))*sum_term
    return cost
    # raise NotImplementedError()

## Now we can plot the lines over iterations
## To do this, we start by constructing a grid of parameter pairs and their corresponding cost function values. 
x_axis = np.linspace(-1,1,100)
theta_grid = torch.Tensor(len(x_axis),2)
theta_grid[:,0] = torch.from_numpy(x_axis)
theta_grid[:,1] = 2.0
# print (theta_grid.t().shape)
# print (X.shape)
# print (y.shape)

J_grid = cost_func(theta_grid.t(), X, y)
print (J_grid.shape)

N = 5
eta = 0.03

theta_0 = torch.Tensor([[0.0], [2.0]]) #initialise 
J_t = torch.Tensor(1,N)
theta = torch.Tensor(2,1,N)
print(theta)
J_t[:,0] = cost_func(theta_0, X, y)[0]
# print (J_t[:,0])
theta[:,:,0] = theta_0

for j in range(1,N):
    last_theta = theta[:,:,j-1]
    ## Compute the value of this_theta
    # YOUR CODE HERE
    grad = grad_cost_func (last_theta, X, y)
    this_theta = last_theta - eta*grad
    # raise NotImplementedError()
    theta[:,:,j] = this_theta
    J_t[:,j] = cost_func(this_theta,X,y)[0]

    
colors = ['b', 'g', 'm', 'c', 'orange']

## Plot the data 
# YOUR CODE HERE
plt.scatter(X[:,0].numpy(), y.numpy())
print (theta[:,:,1].shape)
print (X.shape)
print (theta[:,:,0])
print (theta[:,:,1])
# print (X)
print (theta[:,:,2])
print (theta[:,:,3])
print (theta[:,:,4])
y0 = X@theta[:,:,0]
plt.plot(X[:,0].numpy(), y0[:,0].numpy(),'c')
y1 = X@theta[:,:,1]
# print (X[:,0].shape)
# print (y1)
plt.plot(X[:,0].numpy(), y1[:,0].numpy(),'b')
y2 = X@theta[:,:,2]
# print (y2[26,0])
plt.plot(X[:,0].numpy(), y2[:,0].numpy(),'orange')
y3 = X@theta[:,:,3]
# print (y3[84,0])
plt.plot(X[:,0].numpy(), y3[:,0].numpy(),'g')
y4 = X@theta[:,:,4]
plt.plot(X[:,0].numpy(), y4[:,0].numpy(),'m')
plt.show()


plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Data and fit')