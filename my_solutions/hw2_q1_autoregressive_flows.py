from pytorch_lightning import LightningModule, Trainer

from deepul.hw2_helper import *
from torch.utils.data import Dataset, DataLoader
import torch
from torch.distributions import normal
# import deepul.pytorch_util as ptu

CUDA = torch.cuda.is_available()


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

        self.inner_net = torch.nn.Linear(2, 1)

    def forward(self, inp):
        probs = self.get_probs(inp)
        return - probs.log().mean()

    def get_probs(self, inp):
        z1, z2, m1, m2 = self.get_outputs(inp)

        mask = (z1>=0)&(z1<=1)
        m1 = torch.where(mask, m1, torch.tensor(1e-8))

        mask = (z2>=0)&(z2<=1)
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

    def get_latent_variables(self, inp):
        with torch.no_grad():
            z1, z2, m1, m2 = self.get_outputs(inp)
            z1 = z1.cpu().numpy()
            z2 = z2.cpu().numpy()
            return np.stack([z1, z2], axis=1)


class AutFlow2DEstimator(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoregressiveFlow2D()
        self.losses = []
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        preds = self.model(batch)
        loss = preds.mean()
        self.losses.append(loss.detach().cpu().numpy().item())
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch)
        loss = preds.mean()
        self.log('val/loss', loss, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def validation_epoch_end(self, outputs):
        ll = outputs
        loss = torch.stack(ll).mean()
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.test_losses.append(loss.detach().cpu().numpy().item())
        return loss


def pl_training_loop(train_data, test_data, dset_id):
    global train_losses, test_losses, densities, latents, model

    batch_size = 32
    epochs = 100

    train_ds = Pairs(train_data)
    test_ds = Pairs(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    estimator = AutFlow2DEstimator()
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

    #heatmap
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
        densities = probs[:, 0]*probs[:, 1]
    # densities = np.exp(ptu.get_numpy(ar_flow.log_prob(mesh_xs)))

    # latents

    return train_losses, test_losses, densities, latents


if __name__ == '__main__':
    q1_save_results(1, 'a', q1_a)

    # train_data, train_labels, test_data, test_labels = q1_sample_data_1()

    # inp = torch.rand((6, 2))
    #
    # net = AutoregressiveFlow2D()
    #
    # net(inp)
