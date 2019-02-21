# The standard way to find the pseudoinverse is using Singular Value Decomposition

# X =    U(Sigma)(V^T)
# X^-1 = V(Sigma^-1)(U^T)
import torch
import numpy as np

N = 1000
X = torch.rand(N,2)
X[:, 1] = 1.0

theta_true = torch.Tensor([[1.5], [2.0]])
y = X @ theta_true + 0.1 * torch.randn(N, 1) # Note that just like in numpy '@' represents matrix multiplication and A@B is equivalent to torch.mm(A, B)

u, s, v = torch.svd(X)

X_inv_svd = v@torch.diag(torch.reciprocal(s))@u.t()
theta_pinv_svd = torch.mm(X_inv_svd, y)
print(theta_pinv_svd)
