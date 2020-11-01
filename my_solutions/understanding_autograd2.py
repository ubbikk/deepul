import torch
import torch.autograd
import torch.autograd.functional as F

x = torch.tensor(1., dtype=torch.float, requires_grad=True)
y = torch.tensor(2., dtype=torch.float, requires_grad=True)

def func(x,y):
    return y * (x ** 3) + x * (y ** 2)

j = F.jacobian(func, inputs=(x,y), create_graph=True)

