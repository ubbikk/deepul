import torch

a = torch.tensor(5., requires_grad=True)

b = a.detach()

z = a**2
z.backward()

print(a.grad)
print(b)