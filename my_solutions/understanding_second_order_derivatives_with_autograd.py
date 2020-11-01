import torch
import torch.autograd.functional as F

x = torch.tensor(1., dtype=torch.float, requires_grad=True)
y = torch.tensor(2., dtype=torch.float, requires_grad=True)

z = y * (x ** 3) + x * (y ** 2)

def func(x,y):
    return y * (x ** 3) + x * (y ** 2)

# h = F.hessian(func, inputs=(x,y))
j = F.jacobian(func, inputs=(x,y))

# dz/dx = 3yx^2+y^2
# d^2z/dxdy = 3x^2+2y


# a = torch.tensor([3, 1], dtype=torch.float, requires_grad=True)
# b = a**2
#
# c = torch.tensor([1,0.01], requires_grad=True)
#
# b.backward(c)

