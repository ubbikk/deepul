from pytorch_lightning import LightningModule, Trainer
from torch.utils.tensorboard import SummaryWriter

from deepul.hw2_helper import *
from torch.utils.data import Dataset, DataLoader
import torch
from torch.distributions import normal

# import deepul.pytorch_util as ptu

CUDA = torch.cuda.is_available()


def seed_everything(seed=0):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Pairs(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class NonNegativeLinear(torch.nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight ** 2, self.bias)


class AutoregressiveFlow2D(torch.nn.Module):
    def __init__(self, mixture_dim=50):
        super().__init__()
        self.mixture_dim = mixture_dim
        self.mixture1 = NonNegativeLinear(1, mixture_dim)
        self.mixture1_weights = torch.nn.Parameter(torch.ones(mixture_dim, dtype=torch.float))

        self.mixture2 = NonNegativeLinear(1, mixture_dim)
        self.mixture2_weights = torch.nn.Parameter(torch.ones(mixture_dim, dtype=torch.float))

        self.inner_net = torch.nn.Linear(2, 1)

    def forward(self, inp):
        probs = self.get_probs(inp)
        return - probs.log().mean()

    def get_probs(self, inp):
        z1, z2, m1, m2 = self.get_outputs(inp)

        mask = (z1 >= 0) & (z1 <= 1)
        m1 = torch.where(mask, m1, torch.tensor(1e-8))

        mask = (z2 >= 0) & (z2 <= 1)
        m2 = torch.where(mask, m2, torch.tensor(1e-8))

        return torch.stack([m1, m2], dim=1)

    def get_outputs(self, inp):
        inp = inp.float()

        x1 = inp[:, 0].unsqueeze(-1)
        x2 = inp[:, 1].unsqueeze(-1)

        X = x1
        w = torch.softmax(self.mixture1_weights, dim=0)

        sig = self.mixture1(X)
        sig = torch.sigmoid(sig)
        z1 = (sig @ w).detach()

        sig = sig - sig ** 2
        sig = sig * self.mixture1.weight.T.abs() ** 2

        m1 = sig @ w

        X = self.inner_net(inp)
        w = torch.softmax(self.mixture2_weights, dim=0)

        sig = self.mixture2(X)
        sig = torch.sigmoid(sig)
        z2 = (sig @ w).detach()

        sig = sig - sig ** 2
        sig = sig * self.mixture2.weight.T.abs() ** 2
        sig = sig * self.inner_net.weight[0][1].abs()

        m2 = sig @ w

        return z1, z2, m1, m2

    def get_latent_variables(self, inp):
        with torch.no_grad():
            z1, z2, m1, m2 = self.get_outputs(inp)
            z1 = z1.cpu().numpy()
            z2 = z2.cpu().numpy()
            return np.stack([z1, z2], axis=1)


class AutoregressiveFlow2DQuadraticInnerNet(torch.nn.Module):
    def __init__(self, mixture_dim=50):
        super().__init__()
        self.mixture_dim = mixture_dim
        self.mixture1 = NonNegativeLinear(1, mixture_dim)
        self.mixture1_weights = torch.nn.Parameter(torch.ones(mixture_dim, dtype=torch.float))

        self.mixture2 = NonNegativeLinear(1, mixture_dim)
        self.mixture2_weights = torch.nn.Parameter(torch.ones(mixture_dim, dtype=torch.float))

        self.inner_net = torch.nn.Linear(9, 1)

    def forward(self, inp):
        probs = self.get_probs(inp)
        return - probs.log().mean()

    def get_probs(self, inp):
        z1, z2, m1, m2 = self.get_outputs(inp)

        mask = (z1 >= 0) & (z1 <= 1)
        m1 = torch.where(mask, m1, torch.tensor(1e-8))

        mask = (z2 >= 0) & (z2 <= 1)
        m2 = torch.where(mask, m2, torch.tensor(1e-8))

        return torch.stack([m1, m2], dim=1)

    def get_outputs(self, inp):
        inp = inp.float()

        x1 = inp[:, 0].unsqueeze(-1)
        x2 = inp[:, 1].unsqueeze(-1)

        X = x1
        w = torch.softmax(self.mixture1_weights, dim=0)

        sig = self.mixture1(X)
        sig = torch.sigmoid(sig)
        z1 = (sig @ w).detach()

        sig = sig - sig ** 2
        sig = sig * self.mixture1.weight.T.abs() ** 2

        m1 = sig @ w

        data = [x1, x1 ** 2, x1 ** 3,
                x2, x2 * x1, x2 * x1 ** 2, x2 * x1 ** 3,
                x2 * torch.sin(x1),
                x2*torch.exp(x1)
                ]
        X = torch.stack(data, dim=2).squeeze()
        X = self.inner_net(X)
        w = torch.softmax(self.mixture2_weights, dim=0)

        sig = self.mixture2(X)
        sig = torch.sigmoid(sig)
        z2 = (sig @ w).detach()

        sig = sig - sig ** 2
        sig = sig * self.mixture2.weight.T.abs() ** 2

        inner_weight = self.inner_net.weight
        d_inner = inner_weight[0, 3] + inner_weight[0, 4] * x1 + \
                  inner_weight[0, 5] * x1 ** 2 + inner_weight[0, 6] * x1 ** 3 \
                  + inner_weight[0, 7] * torch.sin(x1)+ inner_weight[0, 8] * torch.exp(x1)
        sig = sig * d_inner.abs()

        m2 = sig @ w

        return z1, z2, m1, m2

    def get_latent_variables(self, inp):
        with torch.no_grad():
            z1, z2, m1, m2 = self.get_outputs(inp)
            z1 = z1.cpu().numpy()
            z2 = z2.cpu().numpy()
            return np.stack([z1, z2], axis=1)


class AutFlow2DEstimator(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoregressiveFlow2D()
        self.losses = []
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        preds = self.model(batch)
        loss = preds.mean()
        self.losses.append(loss.detach().cpu().numpy().item())
        self.log('train/loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch)
        loss = preds.mean()
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def validation_epoch_end(self, outputs):
        ll = outputs
        loss = torch.stack(ll).mean()
        self.test_losses.append(loss.detach().cpu().numpy().item())
        # self.log('val/loss_after_epoch', loss, on_epoch=True, on_step=False)
        self.logger.log_metrics({'val/loss_after_epoch': loss},
                                step=self.trainer.current_epoch)
        return loss


def pl_training_loop(train_data, test_data, dset_id):
    global train_losses, test_losses, densities, latents, model, estimator, trainer

    batch_size = 128
    epochs = 500

    train_ds = Pairs(train_data)
    test_ds = Pairs(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = AutoregressiveFlow2DQuadraticInnerNet()
    estimator = AutFlow2DEstimator(model)
    trainer = Trainer(max_epochs=epochs,
                      gradient_clip_val=1,
                      # gpus=1,
                      # limit_train_batches=3,
                      # limit_val_batches=3,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0
                      # logger=CustomLogger()
                      )
    trainer.fit(estimator,
                train_dataloader=train_loader,
                val_dataloaders=test_loader)

    train_losses = np.array(estimator.losses)
    test_losses = np.array(estimator.test_losses)

    latents = estimator.model.get_latent_variables(torch.tensor(train_data, dtype=torch.float))

    return train_losses, test_losses, latents, estimator.model


model = None
trainer = None
estimator = None
train_losses, test_losses, densities, latents = None, None, None, None


def to_cuda(batch):
    if not CUDA:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def plot_range(model, color, xs=(-2, -1), ys=(2, 3), sz=1000):
    points = torch.rand((sz, 2))
    points[:, 0] = (xs[1] - xs[0]) * points[:, 0] + xs[0]
    points[:, 1] = (ys[1] - ys[0]) * points[:, 1] + ys[0]

    # inp = torch.cartesian_prod(x,y)

    with torch.no_grad():
        z1, z2, _, _ = model.get_outputs(points)

    plt.scatter(z1, z2, color=color, s=1)


def plot_transformation(model, cmap='hsv'):
    x_lim = (-4, 4)
    y_lim = (-4, 4)

    sz = 200
    d = np.array([[x, y] for x in range(sz) for y in range(sz)]) / sz
    d[:, 0] = (x_lim[1] - x_lim[0]) * d[:, 0] + x_lim[0]
    d[:, 1] = (y_lim[1] - y_lim[0]) * d[:, 1] + y_lim[0]

    x, y = d[:, 0], d[:, 1]

    # colors = np.arange(sz**2)/sz**2

    g = 8
    div = sz // g
    # colors = np.array([[x,y] for x in range(sz) for y in range(sz)])
    colors = np.array([g * (x // div) + y // div for x in range(sz) for y in range(sz)])
    colors = colors / float(g ** 2)
    colors = colors.reshape(sz ** 2)

    plt.figure()
    plt.title('Original Space')
    plt.scatter(x, y, c=colors, cmap=cmap, s=1)

    with torch.no_grad():
        z1, z2, _, _ = model.get_outputs(torch.from_numpy(d))

    plt.figure()
    plt.title('Latent Space')
    plt.scatter(z1, z2, c=colors, cmap=cmap, s=1)


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
    global train_losses, test_losses, densities, latents, model

    train_losses, test_losses, latents, model = pl_training_loop(train_data, test_data, dset_id)

    # heatmap
    dx, dy = 0.025, 0.025
    if dset_id == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    elif dset_id == 2:  # two moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = torch.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))
    mesh_xs = to_cuda(mesh_xs)

    with torch.no_grad():
        probs = model.get_probs(mesh_xs)
        densities = probs[:, 0] * probs[:, 1]
    # densities = np.exp(ptu.get_numpy(ar_flow.log_prob(mesh_xs)))

    # latents

    return train_losses, test_losses, densities, latents


if __name__ == '__main__':
    seed_everything()
    q1_save_results(1, 'a', q1_a)

    plt.figure()
    plot_range(model, 'black', (-2, -1), (2, 3))
    plot_range(model, 'green', (1, 2), (2, 3))
    plot_range(model, 'yellow', (-1, 1), (-1, 0))
    # plot_range(model,'red', (-1,1), (0,2))
    plot_range(model, 'red', (-1, 1), (1, 2))
    plot_range(model, 'orange', (-3, -1), (0, 1))
    plot_range(model, 'grey', (-1, 1), (2, 3))

    plt.figure()
    plot_range(model, 'black', (-2, -1), (2, 3), 1000)
    plot_range(model, 'green', (1, 2), (2, 3), 1000)

    plt.figure()
    plot_range(model, 'green', (-3, 3), (0, 1))
    plot_range(model, 'blue', (-3, 3), (1, 2))
    plot_range(model, 'red', (-1, 1), (2, 3))

    plot_range(model, 'yellow', (-3, -2), (2, 3))
    plot_range(model, 'orange', (2, 3), (1, 2))
    # plot_range(model, 'orange', (-3, -1), (0, 1))
    # plot_range(model, 'grey', (-1, 1), (2, 3))

    # plt.figure()
    # plot_range(model,'black', (-1.5, -1), (2,3), 1000)
    # plot_range(model,'green', (1.5,2), (2,3), 1000)

    train_data, train_labels, test_data, test_labels = q1_sample_data_1()

    plt.figure()
    plt.title('Train')
    plt.scatter(train_data[:, 0], train_data[:, 1])

    # inp = torch.rand((6, 2))
    #
    # net = AutoregressiveFlow2D()
    #
    # net(inp)

    print(trainer.logger.log_dir)
