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


def preprocess(x):
    x = x + torch.zeros(x.shape).uniform_(-0.5, 0.5)

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
        self.conv2 = Conv2d(self.n_filters, self.n_filters, (3, 3), stride=1, padding=0)
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

        self.conv1 = Conv2d(self.in_channels, self.n_filters, (3, 3), stride=1, padding=0)
        self.conv2 = Conv2d(self.n_filters, self.out_channels, (3, 3), stride=1, padding=0)

        self.blocks = torch.nn.ModuleList([ResnetBlock(self.n_filters) for _ in range(self.n_blocks)])

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x


class AffineCouplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet = SimpleResnet(in_channels=in_channels, out_channels=2 * self.out_channels)  # ?
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.scale_shift = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, mask):
        s, t = torch.chunk(self.resnet(x * mask), 2, dim=1)
        # calculate log_scale, as done in Q1(b)
        log_scale = self.scale * torch.tanh(s) + self.scale_shift

        t = t * (1.0 - mask)
        log_scale = log_scale * (1.0 - mask)
        z = x * torch.exp(log_scale) + t
        log_det_jacobian = log_scale
        return z, log_det_jacobian

    def invert(self, z, mask):
        with torch.no_grad():
            s, t = torch.chunk(self.resnet(z * mask), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            t = t * (1.0 - mask)
            log_scale = log_scale * (1.0 - mask)

            x = (z - t) * torch.exp(-log_scale)

            return x


def squeeze(inp):
    H = inp.shape[0]
    C = inp.shape[-1]

    a = torch.LongTensor([[1, 2], [3, 4]])
    idx = a.repeat(H // 2, H // 2)

    pp = [inp[idx == i].reshape(H // 2, H // 2, C) for i in range(1, 5)]
    pp = pp[::-1]

    res = torch.stack(pp, dim=2)

    return res


def unsqueeze(x):
    h = x.shape[0]
    c = x.shape[-1]

    H = 2 * h
    C = c // 4
    res = torch.zeros(H, H, C)

    a = torch.LongTensor([[1, 2], [3, 4]])
    idx = a.repeat(H // 2, H // 2)
    idx = torch.stack([idx] * 3, dim=-1)

    res[idx == 4] = x[..., :3].flatten()
    res[idx == 3] = x[..., 3:6].flatten()
    res[idx == 2] = x[..., 6:9].flatten()
    res[idx == 1] = x[..., 9:].flatten()

    return res


def test_squeeze():
    torch.manual_seed(0)

    inp = torch.rand(8, 8, 3)

    H = inp.shape[0]
    c = inp.shape[-1]

    a = torch.LongTensor([[1, 2], [3, 4]])
    idx = a.repeat(H // 2, H // 2)

    pp = [inp[idx == i].reshape(H // 2, H // 2, c) for i in range(1, 5)]
    pp = pp[::-1]

    res = torch.cat(pp, dim=2)


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

    """ YOUR CODE HERE """


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


def mask_for_checkboard_coupling(H=32, white=True):
    idx = torch.arange(H ** 2).reshape(H, H)
    x = idx // H
    y = idx % H

    mask = (x + y) % 2

    mask = mask.float()

    if white:
        return 1 - mask
    else:
        return mask


if __name__ == '__main__':
    os.chdir('/home/ubik/projects/')
    seed_everything(1)
    # q3_save_results(q3_a, 'a')
    train_data, test_data = load_pickled_data('deepul/homeworks/hw2/data/celeb.pkl')
