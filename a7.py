from neural import *

# # Question 1 Below
net_1 = NeuralNet(2, 2, 1)
data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
net_1.train(data)
print()

# # Question 2 Below
net_2 = NeuralNet(2, 8, 1)
data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
net_2.train(data)
print()

# # Question 3 Below
net_3 = NeuralNet(2, 1, 1)
data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
net_3.train(data)
print()

# Question 4 Below
net_4 = NeuralNet(5, 6, 1)
train_data = [([0.9, 0.6, 0.8, 0.3, 0.1], [1]), ([0.8, 0.8, 0.4, 0.6, 0.4], [1]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0])]
net_4.train(train_data)
print()
test_data = [[1, 1, 1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.7, 0.7], [0.8, 0.3, 0.3, 0.3, 0.8], [0.8, 0.3, 0.3, 0.8, 0.3], [0.9, 0.8, 0.8, 0.3, 0.6]]
print(net_4.test(test_data))