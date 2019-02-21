##Norms##

# Norms are function used to measure the size of vectors
# Norm is a measure of the length of the length


# ||x||_p = ((Sum of all values in the vector)^p)^1/p

# THe most common ones are l=1,2,infinity


def linear_regression_loss_grad(theta, X, Y):

        #From the above equation...

        second_term = X@theta - Y
        grad = X.t()@second_term
        return grad

