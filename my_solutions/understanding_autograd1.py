import torch
import torch.autograd

a = torch.tensor(10., requires_grad=True)

b = a**2 + a

c = 3*b+1


x = torch.autograd.grad(c, b, retain_graph=True)[0]
y = torch.autograd.grad(c, a, retain_graph=True)[0]

c.backward()

print(a.grad)