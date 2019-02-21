import torch
import matplotlib.pyplot as plt
from GB_linear import linear_regression_loss_grad

#We are creating some points on a straight line with some Gaussian noise
N = 1000


# Tensor -
#Array of numbers arranged on a regular grid with a variable number of axes

#We are going to make a random tensor - 2,1

theta_true = torch.Tensor([[1.5], [2.0]])

X = torch.rand(N,2)
#Produces 1000x2 vector with values between 0 and 1 taken from normal dist.

#So, this is like:
# [a1, a2]
# [a3, a4]
#     .
#     .
#     .
#    x1000

# All the numbers in the second column a2,a4,...,a1000 is 1 now
X[:, 1] = 1.0

#So, y = 1.5x + 2
Y = X@theta_true + (0.1*torch.randn(N, 1))
# Shape of Y is [1000, 1]
# Random noise is added 0.1 * [1000, 1] of values between 0 and 1

# Now we are plotting the first column of X. Remember the second column was just 1st
# Plot the x values and correspoing Y values from Line 30
# plt.scatter(X[:,0].numpy(), Y.numpy())
# plt.show()

# The problem can be modelled as y = X0 --> 0 = (X^-1)y
# 0 can be considered as a tensor values in Line 13
X_inv = torch.pinverse(X) # This is the pseduoinverse
theta_pinv = torch.mm(X_inv, Y)
# theta_pinv2 = X_inv@Y --> They are the same
print(theta_pinv)

grad = linear_regression_loss_grad(theta_true, X, Y)

alpha = 0.001
theta = torch.Tensor([[0], [0]])
for e in range(0, 200):
    gr = linear_regression_loss_grad(theta, X, Y)
    theta -= alpha * gr

print(theta)
