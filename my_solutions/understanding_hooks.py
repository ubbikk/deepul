import torch

bl = {}


def get_hook(name):
    def hook(grad):
        bl[name] = grad

    return hook


x = torch.Tensor([1]).float()  # x.grad = [2]
x.requires_grad = True

y = torch.Tensor([10]).float()  # y.grad = [3]
y.requires_grad = True

x_inter = x ** 2
x_inter.register_hook(get_hook('x_inter'))

y_inter = 3 * y
y_inter.register_hook(get_hook('y_inter'))

z = 100 * x_inter + y_inter  # z = 130

z[0].backward()

x.grad, y.grad

bl['x_inter']  # x_inter.grad = d(x_inter)/dx = 2, d(x_inter)/dy = 0
bl['y_inter']
