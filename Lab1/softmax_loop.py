import torch

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
    raise NotImplementedError()

    return grad

def denominator(Theta, X, K):

    Sum = 0
    for j in range(K):
        Sum += Theta[:,j].resize_((2,1)).t()@X


    return Sum

def indicator_function(y_i, k):
    if(y_i == k):
        return 1
    else:
        return 0


def softmax_regression_loss(Theta, X, y):
    '''Implementation of the softmax loss function.

    Theta is the matrix of parameters, with the parameters of the k-th class in the k-th column
    X contains the data vectors (one vector per row)
    y is a column vector of the targets
    '''
    # YOUR CODE HERE


    m = X.shape[1] # Should be 13
    print(X.shape)
    K = max(y)

    print(K)

    Loss_Sum = 0;

    for i in range(0,m):
        sum_of_labels = denominator(Theta, X[i,:], K)
        print("SUM OF LABELS VALUE" ,sum_of_labels)
        for k in range(0,K):
            if(indicator_function(y[i], k) == 1):
                exp_term = torch.exp(Theta[:,k].resize_((2,1)).t()@X[i,:].resize_((2,1)))
                print("exp_term: ", exp_term.shape)
                fraction_term = torch.log(exp_term/sum_of_labels)
                Loss_Sum += fraction_term
            else:
                continue


    Loss_Sum = -Loss_Sum
    print("Loss_Sum",Loss_Sum.shape)
    #raise NotImplementedError()

    return Loss_Sum

Theta = torch.Tensor([[1, 0], [0, 1]])
X = torch.Tensor([[1, 0], [0, 1]])
y = torch.LongTensor([[0], [1]])

print("final: ", torch.abs(softmax_regression_loss(Theta, X, y) - 0.6265))
#assert torch.abs(softmax_regression_loss(Theta, X, y) - 0.6265) <
