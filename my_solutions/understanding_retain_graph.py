import torch

bl = []
# def save_grad():
#     def hook(grad):
#         bl.append(grad)
#     return hook

def hook(grad):
    bl.append(grad)

x = torch.ones((3, 2), requires_grad=True)
y = 10 * torch.ones((3, 2))
y.requires_grad = True



x_inter = x ** 2
x_inter.register_hook(hook)
y_inter = 2 * y

z = x_inter + y_inter

# z[0][0].backward()
z[0][0].backward(retain_graph=True)
# z[0][0].backward(retain_variables=True)
