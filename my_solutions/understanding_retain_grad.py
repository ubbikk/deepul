import torch

a=torch.tensor(1., requires_grad=True)
b=a**3
c = 2*b+1

c.backward()
print(a.grad, b.grad)

a=torch.tensor(1., requires_grad=True)
b=a**3
b.retain_grad()
c = 2*b+1

c.backward()
print(a.grad, b.grad)