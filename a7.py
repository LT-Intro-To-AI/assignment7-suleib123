from neural import *

# Question 1 Below
net_1 = NeuralNet(2, 2, 1)
data = [(0, 0), (1, 1), (1, 1), (2, 0)]
net_1.train(data)