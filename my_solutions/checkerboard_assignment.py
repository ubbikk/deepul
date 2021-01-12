import torch

torch.manual_seed(0)

# H=4
# idx = torch.arange(H ** 2).reshape(H, H)
# x = idx // H
# y = idx % H
#
# mask = (x + y) % 2
# mask = mask.bool()

a = torch.arange(50).reshape(5, 10).float()
a.requires_grad=True
b = 100*torch.ones(3)
b.requires_grad=True

c = torch.ones(1).float()
c.requires_grad=True
a[:, :]=c

a[[0,1,2], [4,5,6]] = b