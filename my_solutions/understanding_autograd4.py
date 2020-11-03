import torch
import torch.autograd

a = torch.tensor(10., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

loss = (b**2)*(a**2)


# if you need the gradient wrt a specific variable, you can use
d_a =torch.autograd.grad(loss, a, create_graph=True)[0]
d_ab = torch.autograd.grad(d_a, b, retain_graph=True)[0]

loss.backward()