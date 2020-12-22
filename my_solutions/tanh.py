import matplotlib.pyplot as plt
import torch

a = torch.arange(-20, 20).float()

sigm = torch.sigmoid(a)
tnh = torch.tanh(a)

plt.plot(a, sigm)
plt.plot(a, tnh)
plt.legend(['Sigmoid' , 'Tanh'])