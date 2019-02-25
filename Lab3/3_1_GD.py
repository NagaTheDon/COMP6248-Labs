import torch
import matplotlib.pyplot as plt

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
