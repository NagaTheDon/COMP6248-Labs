import torch
from sklearn.datasets import load_digits
# we wouldn't normally do this, but for this lab we want to work in double precision
# as we'll need the numerical accuracy later on for doing checks on our gradients:
torch.set_default_dtype(torch.float64)

def softmax_regression_loss_grad(Theta, X, y):

    num_examples = y.shape[0]
    num_classes = y.max() + 1

    one_shot = torch.zeros((num_classes, num_examples))
    
    for i in range(num_examples):
        one_shot[y[i,0],i] = 1

    theta_x = torch.mm(X, Theta)
    exp_term = torch.exp(theta_x) # Numerator
    exp_sum = torch.sum(exp_term, (0)) # Denominator 
    # NOTE: The sum is added along the column since the sigma is across the classes

    frac_term = exp_term/exp_sum # This is basically the probability

    subtracted_term = one_shot - frac_term

    grad = -(X.double()*subtracted_term.double())

    grad = torch.sum(grad, (1)) # Maybe 0
    return grad





def softmax_regression_loss(Theta, X, y):

    #One-shot encoding
    
    num_examples = y.shape[0]
    num_classes = y.max() + 1

    one_shot = torch.zeros((num_classes, num_examples))
    
    for i in range(num_examples):
        one_shot[y[i,0],i] = 1

    theta_x = torch.mm(X, Theta)
    exp_term = torch.exp(theta_x) # Numerator
    exp_sum = torch.sum(exp_term, (0)) # Denominator 
    # NOTE: The sum is added along the column since the sigma is across the classes

    frac_term = exp_term/exp_sum 

    cost_matrix = one_shot.double()*torch.log(frac_term).double()
    #NOTE: WE WANT MULTIPLY not dot or mm. We want the diagnols to be zeros

    loss = -(torch.sum(cost_matrix))



    return loss


Theta = torch.Tensor([[1, 0], [0, 1]])
X = torch.Tensor([[1, 0], [0, 1]])
y = torch.LongTensor([[0], [1]])
assert torch.abs(softmax_regression_loss(Theta, X, y) - 0.6265) < 0.0001

grad = softmax_regression_loss_grad(Theta, X, y)
assert torch.torch.allclose(torch.abs(grad/0.2689), torch.ones_like(grad), atol=0.001)