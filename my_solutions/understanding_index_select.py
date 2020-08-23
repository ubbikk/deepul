import torch

a = torch.arange(24).reshape((6, 4))

index = torch.LongTensor([2, 2, 0, 0, 1, 1])

b = torch.gather(a, 1, index)
