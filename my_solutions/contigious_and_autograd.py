import torch

a = torch.ones((3, 4), requires_grad=True)
b = torch.arange(12).reshape((3, 4)).float()
b.requires_grad = True

a_x = a[:, 0]
print(a_x)

b_x = b[:, 0]
print(b_x)

a_x = a_x.contiguous()

z = (2 * a_x + b_x).sum()
print(z)

z.backward()

print(a.grad)

