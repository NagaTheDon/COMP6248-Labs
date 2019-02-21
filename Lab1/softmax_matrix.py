import torch
from sklearn.datasets import load_digits
# we wouldn't normally do this, but for this lab we want to work in double precision
# as we'll need the numerical accuracy later on for doing checks on our gradients:
torch.set_default_dtype(torch.float64)

def softmax_regression_loss_grad(Theta, X, y):
    '''Implementation of the gradient of the softmax loss function.

    Theta is the matrix of parameters, with the parameters of the k-th class in the k-th column
    X contains the data vectors (one vector per row)
    y is a column vector of the targets
    '''
    # YOUR CODE HERE
    # raise NotImplementedError()

    return grad


def denominator(Theta, X, ):

    # product = X@Theta
    # exp = torch.exp(product)
    # print(exp)
    # ans = torch.sum(exp)
    # print("denominaotr: ", ans)
    denominator = 0

    for j in range(Theta.shape[1]):
        denominator += torch.exp(Theta.t()[j]@X[i])

    print("denominator: ", denominator)
    ans = 0
    return ans


def softmax_regression_loss(Theta, X, y):
    sum_of_labels = denominator(Theta, X)

        # num_labels = y.shape[0]
    loss = 0

    num_classes = torch.max(y)+1
    num_examples = y.shape[0]

    one_shot = torch.zeros((num_classes, num_examples))

    for i in range(num_examples):
        one_shot[y[i,0],i] = 1

    print(one_shot)

    log_term = torch.log(torch.exp(X@Theta)/sum_of_labels)

    loss_array = one_shot@log_term

    print(loss_array.shape)

    print(loss_array)

    '''Implementation of the softmax loss function.
    Theta is the matrix of parameters, with the parameters of the k-th class in the k-th column
    X contains the data vectors (one vector per row)
    y is a column vector of the targets
    '''
    # YOUR CODE HERE


    # raise NotImplementedError()

    return loss



# num_classes = 10
# features_dim = 20
# num_items = 100
# Theta = torch.randn((features_dim, num_classes))
# X = torch.randn((num_items,features_dim))
# y = torch.torch.randint(0, num_classes, (num_items, 1))
# # print(y)

# # print("Theta Shape: ", Theta.shape) # 20, 10
# # print("X: ", X.shape) # 100, 20
# # print("Y: ", y.shape) # 100, 1

# grad = softmax_regression_loss(Theta, X, y)
# X*Theta = 100, 10
# Hot shot = 10, 100


Theta = torch.Tensor([[1, 0], [0, 1]])
X = torch.Tensor([[1, 0], [0, 1]])
y = torch.LongTensor([[0], [1]])
assert torch.abs(softmax_regression_loss(Theta, X, y) - 0.6265) < 0.0001
