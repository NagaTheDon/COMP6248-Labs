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

# ## visualise the data by plotting it
# # YOUR CODE HERE
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
    sum_term = (X.t()@(hypothesis(theta, X) - y))
    grad = (1/M)*sum_term
    return grad
    # raise NotImplementedError()

## cost_func computes the cost function J
def cost_func(theta, X, y):
    # YOUR CODE HERE
    sum_term = ((hypothesis(theta, X) - y)**2).sum(0)
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

# Need to fix the problem here 
theta_0 = theta[:,:,0]
theta_0_r = theta_0[0,:]
# print (theta_0_r)
print("!!!!!!!!!!!!!!!!!!")
theta_1 = theta[:,:,1]
theta_1_r = theta_1[0,:]
theta_2 = theta[:,:,2]
theta_2_r = theta_2[0,:]
theta_3 = theta[:,:,3]
theta_3_r = theta_3[0,:]
theta_4 = theta[:,:,4]
theta_4_r = theta_4[0,:]

print("'#''''''''")

print(theta_0_r,theta_2_r, theta_3_r, theta_4_r)

print("'#''''''''")
# print (theta_1_r)
print (J_t[:,0])
print(J_t[:,1])
print(J_t[:,2])
print(J_t[:,3])
print(J_t[:,4])
print(J_grid.shape)
print(theta_grid[:,0].shape)


plt.plot(theta_grid[:,0].numpy(), J_grid.numpy(), c='black')
plt.scatter(theta_0_r.numpy(), J_t[:,0].numpy(),c='c')
plt.scatter(theta_1_r.numpy(), J_t[:,1].numpy(),c='b')
plt.scatter(theta_2_r.numpy(), J_t[:,2].numpy(),c='orange')
plt.scatter(theta_3_r.numpy(), J_t[:,3].numpy(),c='g')
plt.scatter(theta_4_r.numpy(), J_t[:,4].numpy(),c='m')



# add the plot axes labels and title
axes = plt.gca()

plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$J(\theta_1)$')
plt.title('Cost function')
plt.show()

theta_0_vals = np.linspace(-1.0,1,100)
theta_1_vals = np.linspace(-4.0,4,100)

J = torch.zeros(100,100)

for zero_ind in range(100):
    for first_ind in range(100):
        #print(zero_ind, first_ind)
        single_theta = torch.Tensor([[theta_0_vals[zero_ind]],
                               [theta_1_vals[first_ind]]])
        # print(single_theta.shape)

        J[first_ind,zero_ind] = cost_func(single_theta, X, y)# Might need to change this... 

xc,yc = np.meshgrid(theta_0_vals, theta_1_vals)
contours = plt.contour(xc, yc, J, 20)
axes = plt.gca()
axes.set_xlim([-1,1])
axes.set_ylim([-5,5])
plt.clabel(contours)
plt.show()