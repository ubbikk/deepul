import torch

if __name__ == '__main__':
    space = torch.linspace(0, 1, 11)
    space = space.repeat(5, 1)  # (5, 11)

    z = torch.zeros(5, 3)
    z.uniform_()

    idx = torch.searchsorted(space, z)

    space = space[0]
    x = space[idx]

# [0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000]
#
# [0.1671, 0.4323, 0.6109],
# [0.2274, 0.4215, 0.6713],
# [0.1895, 0.3351, 0.5905],
# [0.5608, 0.9487, 0.6064],
# [0.9913, 0.6157, 0.3680]
#
# [ 2,  5,  7],
# [ 3,  5,  7],
# [ 2,  4,  6],
# [ 6, 10,  7],
# [10,  7,  4]
