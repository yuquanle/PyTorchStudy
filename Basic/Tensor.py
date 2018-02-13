import torch

# Construct a 5x3 matrix, uninitialized:
# x = torch.Tensor(5, 3)
# print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
# print(x)

# Get its size
#print(x.size())

# Operations
y = torch.rand(5, 3)

# addition
# print(x + y)
# print(torch.add(x, y))
# # anoutput tensor as argument
# result = torch.Tensor(5, 3)
# torch.add(x, y, out=result)
# print(result)
# # in-place (adds x to y)
# y.add_(x)
# print(y)

# indexing
# print(x)
# print(x[1,:])

# resize/reshape
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)
#
# print(x)
# print(y)
# print(z)
# print(x.size(), y.size(), z.size())

# convert
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)

