import torch
import torch.autograd
import torch.autograd.functional as F

a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2.)
c = a * a * b
dc_da = torch.autograd.grad(c, a, create_graph=True)[0]
d = (a + dc_da) * b
dd_da = torch.autograd.grad(d, a)[0]

print('c: {}, dc_da: {}, d: {}, dd_da: {}'.format(c, dc_da, d, dd_da))



