#author:yuquanle
#data:2018.2.5
#Study of SGD


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# any random value
w = 1.0

# forward pass
def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

# compute gradient
def gradient(x, y):
    return 2*x*(x*w - y)

# Before training
print("predict (before training)", 4, forward(4))

# Training loop
for epoch in range(10):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print("\t grad: ",x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, l)

# After training
print("predict (after training)", 4, forward(4))

