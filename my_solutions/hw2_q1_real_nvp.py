import torch
from pytorch_lightning import LightningModule, Trainer
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

from deepul.hw2_helper import *

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


class SimpleCouplingLayer2D(torch.nn.Module):
    def __init__(self, inplace=0):
        super().__init__()
        self.inplace = inplace
        self.scale = torch.nn.Parameter(torch.zeros(1))
        self.scale_shift = torch.nn.Parameter(torch.zeros(1))
        self.g_scale = torch.nn.Parameter(torch.zeros(1))
        self.g_shift = torch.nn.Parameter(torch.zeros(1))

        self.scale.data.normal_(mean=0.0, std=0.02)
        self.scale_shift.data.normal_(mean=0.0, std=0.02)
        self.g_scale.data.normal_(mean=0.0, std=0.02)
        self.g_shift.data.normal_(mean=0.0, std=0.02)

    def forward(self, inp):
        x1, x2 = torch.chunk(inp, chunks=2, dim=1)

        if self.inplace == 0:
            z1 = x1
            log_scale = self.scale * torch.tanh(self.g_scale * x1) + self.scale_shift
            z2 = torch.exp(log_scale) * x2 + self.g_shift

            return torch.cat([z1, z2], dim=1), log_scale
        else:
            z2 = x2
            log_scale = self.scale * torch.tanh(self.g_scale * x2) + self.scale_shift
            z1 = torch.exp(log_scale) * x1 + self.g_shift

            return torch.cat([z1, z2], dim=1), log_scale

    def invert(self, inp):
        with torch.no_grad():
            z1, z2 = torch.chunk(inp, chunks=2, dim=1)
            if self.inplace == 0:
                x1 = z1
                log_scale = self.scale * torch.tanh(self.g_scale * x1) + self.scale_shift
                x2 = (z2 - self.g_shift) * torch.exp(-log_scale)
            else:
                x2 = z2
                log_scale = self.scale * torch.tanh(self.g_scale * x2) + self.scale_shift
                x1 = (z1 - self.g_shift) * torch.exp(-log_scale)

            return torch.cat([x1, x2], dim=1)


class MlpCouplingLayer2D(torch.nn.Module):
    def __init__(self, inplace=0):
        super().__init__()
        self.inplace = inplace
        self.scale = torch.nn.Parameter(torch.zeros(1))
        self.scale_shift = torch.nn.Parameter(torch.zeros(1))
        self.g_scale = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.g_shift = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        self.scale.data.normal_(mean=0.0, std=0.02)
        self.scale_shift.data.normal_(mean=0.0, std=0.02)
        # self.g_scale.data.normal_(mean=0.0, std=0.02)
        # self.g_shift.data.normal_(mean=0.0, std=0.02)

    def forward(self, inp):
        x1, x2 = torch.chunk(inp.float(), chunks=2, dim=1)

        if self.inplace == 0:
            z1 = x1
            log_scale = self.scale * torch.tanh(self.g_scale(x1)) + self.scale_shift
            z2 = torch.exp(log_scale) * x2 + self.g_shift(x1)

            return torch.cat([z1, z2], dim=1), log_scale
        else:
            z2 = x2
            log_scale = self.scale * torch.tanh(self.g_scale(x2)) + self.scale_shift
            z1 = torch.exp(log_scale) * x1 + self.g_shift(x2)

            return torch.cat([z1, z2], dim=1), log_scale


class RealNVP2D(torch.nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        inversions = [0, 1] * (layers // 2)
        self.layers = torch.nn.ModuleList([MlpCouplingLayer2D(i) for i in inversions])
        self.d = Normal(0, 1)

    def forward(self, inp):
        det = 0
        x = inp
        for l in self.layers:
            z, dz = l(x)
            det += dz
            x = z

        loss = det.squeeze()
        loss += self.d.log_prob(x).sum(dim=1)
        loss = loss.mean()
        loss = -loss / 2
        return x, det, loss

    def get_latent_variables(self, inp):
        with torch.no_grad():
            z, dz, loss = self.forward(inp)
            return z

    def get_probs(self, inp):
        z, _, _ = self.forward(inp)
        return self.d.log_prob(z).sum(dim=1).exp()


class AutFlow2DEstimator(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = []
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        _, _, loss = self.model(batch)
        self.losses.append(loss.detach().cpu().numpy().item())
        self.log('train/loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self.model(batch)
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

    batch_size = 64
    epochs = 100

    train_ds = Pairs(train_data)
    test_ds = Pairs(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = RealNVP2D(layers=20)
    estimator = AutFlow2DEstimator(model)
    trainer = Trainer(max_epochs=epochs,
                      gradient_clip_val=1,
                      gpus=int(CUDA),
                      # limit_train_batches=3,
                      # limit_val_batches=3,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      progress_bar_refresh_rate=5 if CUDA else 20,
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
        z, _, _ = model(points)
        z1, z2 = torch.chunk(z, chunks=2, dim=1)

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
        z, _, _ = model(torch.from_numpy(d))
        z1, z2 = torch.chunk(z, chunks=2, dim=1)

    plt.figure()
    plt.title('Latent Space')
    plt.scatter(z1, z2, c=colors, cmap=cmap, s=1)


def q1_b(train_data, test_data, dset_id):
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
        if CUDA:
            model.cuda()
        probs = model.get_probs(mesh_xs)
        densities = probs.cpu()

    return train_losses, test_losses, densities, latents


if __name__ == '__main__':
    seed_everything()
    q1_save_results(1, 'b', q1_b)

    plt.figure()
    plot_range(model, 'black', (-2, -1), (2, 3))
    plot_range(model, 'green', (1, 2), (2, 3))
    plot_range(model, 'yellow', (-1, 1), (-1, 0))
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

    train_data, train_labels, test_data, test_labels = q1_sample_data_1()

    plt.figure()
    plt.title('Train')
    plt.scatter(train_data[:, 0], train_data[:, 1])

    print(trainer.logger.log_dir)

    # model = NormalizingFlowOfCdfs(5)
    # x = torch.rand((16,1))
    # z, dz = model(x)
