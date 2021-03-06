import torch
from pytorch_lightning import LightningModule, Trainer
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn import Conv2d

from deepul.hw2_helper import *

CUDA = torch.cuda.is_available()
torch.autograd.detect_anomaly()


class CelebDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def to_cuda(batch):
    if not CUDA:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def preprocess(x):
    x = x + to_cuda(torch.zeros(x.shape).uniform_(-0.5, 0.5))

    mx = 4
    alfa = 0.05
    x = torch.sigmoid(alfa + (1 - alfa) * (x / mx))

    return x


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


class ResnetBlock(torch.nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters

        self.conv1 = Conv2d(self.n_filters, self.n_filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2d(self.n_filters, self.n_filters, (3, 3), stride=1, padding=1)
        self.conv3 = Conv2d(self.n_filters, self.n_filters, (1, 1), stride=1, padding=0)

    def forward(self, x):
        h = x

        h = self.conv1(h)
        h = torch.relu(h)

        h = self.conv2(h)
        h = torch.relu(h)

        h = self.conv3(h)
        return h + x


class SimpleResnet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=128, n_blocks=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.n_blocks = n_blocks

        self.conv1 = Conv2d(self.in_channels, self.n_filters, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2d(self.n_filters, self.out_channels, (3, 3), stride=1, padding=1)

        self.blocks = torch.nn.ModuleList([ResnetBlock(self.n_filters) for _ in range(self.n_blocks)])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        return x


class AffineCouplingLayer(torch.nn.Module):
    def __init__(self, mask, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet = SimpleResnet(in_channels=in_channels, out_channels=2 * self.out_channels)  # ?
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.scale_shift = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('mask', mask)

    def forward(self, x):
        s, t = torch.chunk(self.resnet(x * self.mask), 2, dim=-1)
        # calculate log_scale, as done in Q1(b)
        log_scale = self.scale * torch.tanh(s) + self.scale_shift

        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        z = x * torch.exp(log_scale) + t
        log_det_jacobian = log_scale
        return z, log_det_jacobian

    def invert(self, z):
        with torch.no_grad():
            s, t = torch.chunk(self.resnet(z * self.mask), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            t = t * (1.0 - self.mask)
            log_scale = log_scale * (1.0 - self.mask)

            x = (z - t) * torch.exp(-log_scale)

            return x


def squeeze(inp):
    B = inp.shape[0]
    H = inp.shape[1]
    C = inp.shape[-1]

    a = torch.LongTensor([[1, 2], [3, 4]])
    idx = a.repeat(H // 2, H // 2)

    pp = [inp[(idx == i).repeat(B, 1, 1)].reshape(B, H // 2, H // 2, C) for i in range(1, 5)]
    pp = pp[::-1]

    res = torch.cat(pp, dim=3)

    return res


def unsqueeze(x):
    b = x.shape[0]
    h = x.shape[1]
    c = x.shape[-1]

    H = 2 * h
    C = c // 4
    res = torch.zeros(b, H, H, C)
    res = to_cuda(res)

    a = torch.LongTensor([[1, 2], [3, 4]])
    idx = a.repeat(H // 2, H // 2)
    idx = torch.stack([idx] * 3, dim=-1)

    mask = (idx == 4).repeat(b, 1, 1, 1)
    mask = to_cuda(mask)
    res[mask] = x[..., :3].flatten()

    mask = (idx == 3).repeat(b, 1, 1, 1)
    mask = to_cuda(mask)
    res[mask] = x[..., 3:6].flatten()

    mask = (idx == 2).repeat(b, 1, 1, 1)
    mask = to_cuda(mask)
    res[mask] = x[..., 6:9].flatten()

    mask = (idx == 1).repeat(b, 1, 1, 1)
    mask = to_cuda(mask)
    res[mask] = x[..., 9:].flatten()

    return res


def test_squeeze_unsqueze():
    torch.manual_seed(0)

    inp = torch.rand(10, 8, 8, 3)
    inp.requires_grad = True

    s = squeeze(inp)
    out = unsqueeze(s)

    print((inp == out).all())
    print(out.requires_grad)


def mask_for_checkerboard_coupling(H=32, C=3, white=True):
    idx = torch.arange(H ** 2).reshape(H, H)
    x = idx // H
    y = idx % H

    mask = (x + y) % 2

    mask = mask.float()

    if white:
        mask = 1 - mask

    return torch.stack([mask] * C, dim=-1)


def mask_for_channel_coupling(h=16, c=12, white=True):
    mask = torch.ones(h, h, c).float()
    if white:
        mask[..., :c // 2] = 0
    else:
        mask[..., c // 2:] = 0

    return mask


class RealNVP(torch.nn.Module):
    def __init__(self, H=32, C=3):
        super().__init__()
        self.H = H
        self.C = C

        masks1 = [mask_for_checkerboard_coupling(H, C, w) for w in [True, False, True]]
        masks2 = [mask_for_channel_coupling(H // 2, C * 4, w) for w in [True, False, True]]
        masks3 = [mask_for_checkerboard_coupling(H, C, w) for w in [False, True, False]]

        self.coupling_layers1 = torch.nn.ModuleList([AffineCouplingLayer(mask, C, C) for mask in masks1])
        self.coupling_layers2 = torch.nn.ModuleList([AffineCouplingLayer(mask, C * 4, C * 4) for mask in masks2])
        self.coupling_layers3 = torch.nn.ModuleList([AffineCouplingLayer(mask, C, C) for mask in masks3])

    def forward(self, x):
        dz = 0

        for l in self.coupling_layers1:
            x, jacobian = l(x)
            dz += jacobian.sum()

        x = squeeze(x)

        for l in self.coupling_layers2:
            x, jacobian = l(x)
            dz += jacobian.sum()

        x = unsqueeze(x)

        for l in self.coupling_layers3:
            x, jacobian = l(x)
            dz += jacobian.sum()

        z = x

        return z, dz

    def invert(self, z):
        with torch.no_grad():
            for l in self.coupling_layers3[::, -1]:
                z, _ = l.invert(z)

            z = squeeze(z)

            for l in self.coupling_layers2[::, -1]:
                z, _ = l.invert(z)

            z = unsqueeze(z)

            for l in self.coupling_layers1[::, -1]:
                z, _ = l.invert(z)

            return z

    def generate_examples(self):
        return torch.randint(0, 2, (100, 32, 32, 2))  # torch.rand(100, 32, 32, 2)


class RealNvpEstimator(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = []
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        batch = preprocess(batch.float())
        _, loss = self.model(batch)
        self.losses.append(loss.detach().cpu().numpy().item())
        self.log('train/loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = preprocess(batch.float())
        _, loss = self.model(batch)
        self.test_losses.append(loss.detach().cpu().numpy().item())
        self.log('val/loss', loss, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def validation_epoch_end(self, outputs):
        ll = outputs
        loss = torch.stack(ll).mean()
        self.test_losses.append(loss.detach().cpu().numpy().item())


model = None
trainer = None
estimator = None
train_losses, test_losses = None, None
examples = None
train_d, test_d = None, None


def pl_training_loop(train_data, test_data):
    global train_losses, test_losses, model, estimator, trainer

    batch_size = 10
    epochs = 5

    train_ds = CelebDataset(train_data)
    test_ds = CelebDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = RealNVP()
    estimator = RealNvpEstimator(model)
    trainer = Trainer(max_epochs=epochs,
                      gradient_clip_val=1,
                      gpus=int(CUDA),
                      # limit_train_batches=3,
                      # limit_val_batches=3,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      progress_bar_refresh_rate=10,
                      )
    trainer.fit(estimator,
                train_dataloader=train_loader,
                val_dataloaders=test_loader)

    train_losses = np.array(estimator.losses)
    test_losses = np.array(estimator.test_losses)

    return train_losses, test_losses, estimator.model


def q3_a(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

    global train_losses, test_losses, model, examples, train_d, test_d

    train_d = train_data
    test_d = test_data

    train_losses, test_losses, model = pl_training_loop(train_data, test_data)
    examples = model.generate_examples()

    interpolations = torch.randint(0, 2, (30, 32, 32, 2))  # torch.rand(30, 32, 32, 2)

    return train_losses, test_losses, examples, interpolations


def visualize(train_data, sz=1):
    idxs = np.random.choice(len(train_data), replace=False, size=(sz,))
    images = train_data[idxs].astype(np.float32) / 3.0 * 255.0
    samples = (torch.FloatTensor(images) / 255).permute(0, 3, 1, 2)
    plt.imshow(samples[0].permute(1, 2, 0))


def logit_smoothing():
    mx = 4
    x = torch.arange(0, mx)
    alfa = 0.05
    y = torch.sigmoid(alfa + (1 - alfa) * (x / mx))
    plt.plot(x, y)
    plt.plot(x[[0, -1]], y[[0, -1]])

    plt.legend(['logit', 'linear'])


# if __name__ == '__main__':
#     os.chdir('/home/ubik/projects/')
#     seed_everything(1)
#     q3_save_results(q3_a, 'a')
#     train_data, test_data = load_pickled_data('deepul/homeworks/hw2/data/celeb.pkl')
#
#     test_squeeze_unsqueze()
