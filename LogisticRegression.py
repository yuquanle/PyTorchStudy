# date:2018.2.6
# LogisticRegression

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[0.6], [1.0], [3.5], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in one out
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

# Our model
model = Model()

# Construct loss function and optimizer
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # Forward pass
    y_pred = model(x_data)

    # Compute loss
    loss = criterion(y_pred, y_data)
    if epoch % 20 == 0:
        print(epoch, loss.data[0])

    # Zero gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update weights
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([[0.5]]))
print("predict (after training)", 0.5, model.forward(hour_var).data[0][0])
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict (after training)", 7.0, model.forward(hour_var).data[0][0])
