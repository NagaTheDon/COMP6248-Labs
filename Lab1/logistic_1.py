import torch

# Predict the probability that a given example belongs to "1" class versus the probability that belongs to "0" class

# Probability of y = 1 given X = \frac{1}{1+exp(\x^T \theta)}
# Probability of y = 0 given X = 1 - Probability of y = 1 given X

# The fraction is called sigmoid function. Squashes any real valued input into range of 0 and 1 which interprets the output as a probability

# Goal: Value of \theta so that the probability is large when x belongs to the "1" class
#               Small when x belongs to the "0" class

