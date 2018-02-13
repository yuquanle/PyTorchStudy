#author:yuquanle
#data:2018.2.5
#Study of LinearRegression use PyTorch

import torch
from torch.autograd import Variable

# train data
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# our model
model = Model()

criterion = torch.nn.MSELoss(size_average=False) # Defined loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Defined optimizer

# Training: forward, loss, backward, step
# Training loop
for epoch in range(51):
    # Forward pass
    y_pred = model(x_data)

    # Compute loss
    loss = criterion(y_pred, y_data)
    if epoch % 5 ==0:
         print(epoch, loss.data[0])

    # Zero gradients
    optimizer.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])