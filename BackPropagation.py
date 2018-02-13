#author:yuquanle
#data:2018.2.6
#Study of BackPagation

import torch
from torch import nn
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# Any random value
w = Variable(torch.Tensor([1.0]), requires_grad=True)

# forward pass
def forward(x):
    return x*w

# Before training
print("predict (before training)", 4, forward(4))

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

# Training: forward, backward and update weight
# Training loop
for epoch in range(10):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\t grad:", x, y, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        # Manually zero the gradients after running the backward pass and update w
        w.grad.data.zero_()
    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)", 4, forward(4))



