import torch
import torch.autograd

a = torch.tensor(10., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

loss = (b**2)*(a**2)

# dl_db = torch.autograd.grad(loss, b, create_graph=True)[0]
# dl_dba = torch.autograd.grad(dl_db, a)[0]



loss.backward(create_graph=True)
g = b.grad
g.backward()
g2 = g.grad

