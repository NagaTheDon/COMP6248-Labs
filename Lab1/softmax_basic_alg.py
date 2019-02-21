import numpy as np

X = np.array([[1, 0], [0, 1]])
Theta = np.array([[1, 0], [0, 1]])
y = np.array([[1,0],[0,1]])

theta_x = np.dot(X, Theta)
hypothesis = np.exp(theta_x)
sum_hyp = np.sum(hypothesis, axis = 0)
print(hypothesis)
print(sum_hyp)

probabilities = hypothesis/sum_hyp
print("=============")
print(probabilities)

cost_examples = np.multiply(y, np.log(probabilities))
print("===============")
print(cost_examples)

print("============")
total_sum = -(np.sum(cost_examples))
print(total_sum)

print(total_sum - 0.6265)

assert (total_sum - 0.6265) < 0.0001
