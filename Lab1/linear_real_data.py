import torch
from GB_linear import linear_regression_loss_grad

# Dataset of house prices in Boston
from sklearn.datasets import load_boston

# load_boston(True) gives data and regression targets
# Target is basically dependent variable ad response variable.
# The thing we are trying to predict.

# In terms of this data set?

# The first one is the data for 506 houses with 13 different features
# THe second one is the house prices of those houses

# The first step is to put in a Tensor

X, Y = tuple(torch.Tensor(z) for z in load_boston(True))

X = X[:,[2,5]]; # Reducing the number of features we are using
#(506,2)
X = torch.cat( (X, torch.ones((X.shape[0],1))) ,1) #Should be (506,3)

Y = Y.reshape(-1, 1) #Should be (506,1)

print("X: ", X.shape)
print("Y: ", Y.shape)

# Training and test data split

perm = torch.randperm(Y.shape[0])
#Creates a tensor of array betweeen 0 - 506 randomly arranged

X_train = X[perm[0:253], :] # Get the values from the data which have index values between 0 and 253
Y_train = Y[perm[0:253]] # Same with Y_train

X_test  = X[perm[253:], :] # The rest goes to the test set
Y_test  = Y[perm[253:]] # Same with Y_test

X_inv_svd = torch.pinverse(X_train)
theta_svd = torch.mm(X_inv_svd, Y_train)
print("Theta: ", theta_svd.t())
print("MSE of test data: ", torch.nn.functional.mse_loss(X_test @ theta_svd, Y_test))

alpha = 0.00001
theta_gd = torch.rand((X_train.shape[1], 1))
for e in range(0, 10000):
    gr = linear_regression_loss_grad(theta_gd, X_train, Y_train)
    theta_gd -= alpha * gr

print("Gradient Descent Theta: ", theta_gd.t())
print("MSE of test data: ", torch.nn.functional.mse_loss(X_test @ theta_gd, Y_test))

# From the equation $$\theta - \alphaf'(\theta)$$, we know that high learning rate will
# increase the value of jumps to reach the lowest loss function.

# Making it less likely to reach the optimal solution. Therefore, it only
# makes sense to decrease the learning rate.
