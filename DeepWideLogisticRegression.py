#author:yuquanle
#date:2018.2.7
#Deep and Wide


import torch
from torch.autograd import Variable
import numpy as np

xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

#print(x_data.data.shape)
#print(y_data.data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.l1(x))
        x = self.sigmoid(self.l2(x))
        y_pred = self.sigmoid(self.l3(x))
        return y_pred

# our model
model = Model()

cirterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

hour_var = Variable(torch.Tensor([[-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333]]))
print("(Before training)", model.forward(hour_var).data[0][0])

# Training loop
for epoch in range(1000):
    y_pred = model(x_data)
    # y_pred,y_data不能写反(因为损失函数为交叉熵loss)
    loss = cirterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(epoch, loss.data[0])


# After training
hour_var = Variable(torch.Tensor([[-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333]]))
print("predict (after training)", model.forward(hour_var).data[0][0])

