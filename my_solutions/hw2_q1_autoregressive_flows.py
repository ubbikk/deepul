from deepul.hw2_helper import *
from torch.utils.data import Dataset, DataLoader
import torch
from torch.distributions import normal

class Pairs(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class AutoregressiveFlow2D(torch.nn.Module):
    def __init__(self, mixture_dim=5):
        super().__init__()
        self.mixture_dim = mixture_dim
        self.mixture1 = torch.nn.Linear(1, mixture_dim)
        self.mixture1_weights = torch.nn.Parameter(torch.ones(mixture_dim, dtype=torch.float))

        self.mixture2 = torch.nn.Linear(1, mixture_dim)
        self.mixture2_weights = torch.nn.Parameter(torch.ones(mixture_dim, dtype=torch.float))

        self.inner_net = torch.nn.Linear(2,1)

    def forward(self, inp):
        z1, z2, m1, m2 = self.get_zs(inp)

        return -(m1.log().sum()+m2.log().sum())/2

    def get_zs(self, inp):
        x1 = inp[:, 0].unsqueeze(-1)
        x2 = inp[:, 1].unsqueeze(-1)

        X = x1
        w = torch.softmax(self.mixture1_weights, dim=0)

        sig = self.mixture1(X)
        sig = torch.sigmoid(sig)
        z1 = (sig @ w).detach()

        sig = sig - sig ** 2
        sig = sig * self.mixture1.weight.T.abs()

        m1 = sig @ w

        X = self.inner_net(inp)
        w = torch.softmax(self.mixture2_weights, dim=0)

        sig = self.mixture2(X)
        sig = torch.sigmoid(sig)
        z2 = (sig @ w).detach()

        sig = sig - sig ** 2
        sig = sig * self.mixture2.weight.T.abs()
        sig = sig * self.inner_net.weight[0][1].abs()

        m2 = sig @ w

        return z1, z2, m1, m2


def q1_a(train_data, test_data, dset_id):
  """
  train_data: An (n_train, 2) numpy array of floats in R^2
  test_data: An (n_test, 2) numpy array of floats in R^2
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets, or
             for plotting a different region of densities

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (?,) of probabilities with values in [0, +infinity).
      Refer to the commented hint.
  - a numpy array of size (n_train, 2) of floats in [0,1]^2. This represents
      mapping the train set data points through our flow to the latent space.
  """

  """ YOUR CODE HERE """
  # create data loaders

  # model

  # train

  # heatmap
  # dx, dy = 0.025, 0.025
  # if dset_id == 1:  # face
  #     x_lim = (-4, 4)
  #     y_lim = (-4, 4)
  # elif dset_id == 2:  # two moons
  #     x_lim = (-1.5, 2.5)
  #     y_lim = (-1, 1.5)
  # y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
  #                 slice(x_lim[0], x_lim[1] + dx, dx)]
  # mesh_xs = ptu.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))
  # densities = np.exp(ptu.get_numpy(ar_flow.log_prob(mesh_xs)))

  # latents

  return train_losses, test_losses, densities, latents


if __name__ == '__main__':
    # q1_save_results(1, 'a', q1_a)

    # train_data, train_labels, test_data, test_labels = q1_sample_data_1()

    inp = torch.rand((6,2))

    net = AutoregressiveFlow2D()

    net(inp)