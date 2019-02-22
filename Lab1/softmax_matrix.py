import torch
from sklearn.datasets import load_digits
from random import randrange

# we wouldn't normally do this, but for this lab we want to work in double precision
# as we'll need the numerical accuracy later on for doing checks on our gradients:
torch.set_default_dtype(torch.float64)

def softmax_regression_loss_grad(Theta, X, y):

    num_examples = y.shape[0]
    num_classes = y.max() + 1

    one_shot = torch.zeros((num_classes, num_examples))

    for i in range(num_examples):
        one_shot[y[i,0],i] = 1

    X_Theta = torch.mm(X,Theta)

    exp_term = torch.exp(X_Theta)
    frac_term = exp_term/torch.sum(exp_term,(0))

    difference = (one_shot.double() - frac_term.t())

    theta_grad = torch.mm(difference, X)

    grad = -(theta_grad.t())
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

    cost_matrix = one_shot.double().t()*torch.log(frac_term).double()
    #NOTE: WE WANT MULTIPLY not dot or mm. We want the diagnols to be zeros

    loss = -(torch.sum(cost_matrix))



    return loss

def grad_check(f, x, analytic_grad, num_checks=10, h=1e-3):
    sum_error = 0

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape]) #randomly sample value to change

        oldval = x[ix].item()
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic) + 1e-8)
        sum_error += rel_error
        print('numerical: %f\tanalytic: %f\trelative error: %e' % (grad_numerical, grad_analytic, rel_error))
    return sum_error / num_checks

# Create some test data:
num_classes = 10
features_dim = 20
num_items = 100
Theta = torch.randn((features_dim, num_classes))
X = torch.randn((num_items,features_dim))
y = torch.torch.randint(0, num_classes, (num_items, 1))

# compute the analytic gradient
grad = softmax_regression_loss_grad(Theta, X, y)

# run the gradient checker    
grad_check(lambda th: softmax_regression_loss(th, X, y), Theta, grad)




Theta = torch.Tensor([[1, 0], [0, 1]])
X = torch.Tensor([[1, 0], [0, 1]])
y = torch.LongTensor([[0], [1]])
assert torch.abs(softmax_regression_loss(Theta, X, y) - 0.6265) < 0.0001

grad = softmax_regression_loss_grad(Theta, X, y)
assert torch.torch.allclose(torch.abs(grad/0.2689), torch.ones_like(grad), atol=0.001)

X, y = (torch.Tensor(z) for z in load_digits(10, True)) #convert to pytorch Tensors
X = torch.cat((X, torch.ones((X.shape[0], 1))), 1) # append a column of 1's to the X's
X /= 255
y = y.reshape(-1, 1) # reshape y into a column vector
y = y.type(torch.LongTensor)

perm = torch.randperm(y.shape[0])
X_train = X[perm[0:260], :]
y_train = y[perm[0:260]]
X_test = X[perm[260:], :]
y_test = y[perm[260:]]

alpha = 0.1
theta_gd = torch.rand((X_train.shape[1], 10))

for e in range(0, 1000):
    gr = softmax_regression_loss_grad(theta_gd, X_train, y_train)
    theta_gd -= alpha * gr
    if e%100 == 0:
        print("Training Loss: ", softmax_regression_loss(theta_gd, X_train, y_train))

# Compute the accuracy of the test set
proba = torch.softmax(X_test @ theta_gd, 1)
print(float((proba.argmax(1)-y_test[:,0]==0).sum()) / float(proba.shape[0]))