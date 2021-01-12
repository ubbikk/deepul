import torch

torch.manual_seed(0)

a = torch.rand(5)
a.requires_grad = True
b = torch.rand(5)
b.requires_grad = True

x = a**2
y = b**3

z = torch.FloatTensor(5)
print(z.requires_grad)

z[:2] = x[:2]
print(z.requires_grad)
z[2:] = y[2:]

z.sum().backward()

print(a.grad)
print(b.grad)


