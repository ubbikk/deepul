from torch.distributions.normal import Normal
import torch

a = Normal(0, 1)

b = Normal(0,2)

# c = a+b

m = torch.tensor([0], dtype=torch.float, requires_grad=True)
s = torch.tensor([1], dtype=torch.float, requires_grad=True)
c = Normal(m,s)
c.icdf()