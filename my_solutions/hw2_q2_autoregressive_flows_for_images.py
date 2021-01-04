import torch
from pytorch_lightning import LightningModule, Trainer
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader

from deepul.hw2_helper import *

CUDA = torch.cuda.is_available()
torch.autograd.detect_anomaly()


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


class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data.transpose(0, 3, 1, 2)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def dequantize(x):
    delta = (1 - 2 * x) * to_cuda(torch.zeros(x.shape).uniform_(0, 0.5))
    return x + delta


def get_conv_mask(sz, type='A'):
    res = np.ones(sz * sz)
    if type == 'A':
        res[sz * sz // 2:] = 0
    else:
        res[1 + sz * sz // 2:] = 0
    return torch.from_numpy(res.reshape(sz, sz)).float()


class MaskedConv2D(torch.nn.Conv2d):
    def __init__(self, mask, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)

        mask = mask.reshape(1, 1, *kernel_size)
        mask = mask.repeat(1, in_channels, 1, 1)
        self.register_buffer('mask', mask)

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PixelCnnFlow(torch.nn.Module):
    def __init__(self, H, W, mixture_dim=5, debug=False):
        super().__init__()
        self.H = H
        self.W = W
        self.mixture_dim = mixture_dim
        self.debug = debug
        maskA = get_conv_mask(7, 'A')
        convA = MaskedConv2D(maskA, in_channels=1, out_channels=64,
                             kernel_size=(7, 7), padding=3)
        maskB = get_conv_mask(7, 'B')
        convsB = [MaskedConv2D(maskB, in_channels=64, out_channels=64, kernel_size=(7, 7), padding=3) for i in
                  range(5)]
        conv1D1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # conv1D2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        convs = [convA] + convsB + [conv1D1]

        self.convs = torch.nn.ModuleList(convs)
        self.fc = torch.nn.Linear(in_features=64, out_features=3 * self.mixture_dim)

        self.basic_distribution = Normal(0, 1)

    def forward(self, *inp):
        x = inp[0].float()
        # x = dequantize(x)
        orig = x.permute(0, 2, 3, 1)

        b, c, H, W = x.shape
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        weight, loc, log_scale = torch.chunk(self.fc(x), chunks=3, dim=-1)

        dist = Normal(loc, log_scale.exp())
        weight = torch.softmax(weight, dim=-1)

        z = (dist.cdf(orig) * weight).sum(dim=-1)
        dz = (dist.log_prob(orig).exp() * weight).sum(dim=-1)
        log_pz = self.basic_distribution.log_prob(z)

        dz+=1e-8

        loss = - dz.log().mean() - log_pz.mean()

        if torch.isnan(loss).tolist() or torch.isinf(loss).tolist():
            print('blja')
            print(loss)

        return z, dz, (loc, log_scale, weight), loss

    def bisection_search(self, z, loc, log_scale, weight):
        """

        :param z: input to invert (B, )
        :param loc: (B, mixture_dim)
        :param log_scale: (B, mixture_dim)
        :param weight: (B, mixture_dim)
        :return: x
        """
        B, mixture_dim = loc.shape
        dist = Normal(loc, log_scale.exp())

        space = torch.linspace(-10, 10, 10001)
        if CUDA:
            space = space.cuda()
        space = space.repeat((B, mixture_dim, 1)).permute(2, 0, 1)  # (1001, B, mixture_dim)

        vals = dist.cdf(space)  # (1001, B, mixture_dim)
        vals = (vals * weight).sum(dim=-1)  # (1001, B)
        vals = vals.transpose(0, 1)  # (B, 1001)

        space = space[:, 0, 0]
        idx = torch.searchsorted(vals, z.reshape(B, 1)).squeeze()
        x = space[idx]

        return x

    def generate_examples(self, sz=100):
        if CUDA:
            self.cuda()
        self.eval()
        with torch.no_grad():
            inp = torch.zeros((sz, self.H, self.W), dtype=torch.float)#.uniform_(0, 1)
            inp = to_cuda(inp)

            Z = self.basic_distribution.sample((sz, self.H, self.W))
            Z = to_cuda(Z)

            for pos in range(self.H * self.W):
                z, _, (loc, log_scale, weight), _ = self(inp.reshape(sz, 1, self.H, self.W))
                i = pos // self.H
                j = pos % self.H
                x = self.bisection_search(Z[:, i, j],
                                          loc[:, i, j, :],
                                          log_scale[:, i, j, :],
                                          weight[:, i, j, :])
                inp[:, i, j] = x

            return inp.reshape((sz, self.H, self.W, 1)).cpu().numpy()


def check_bisection_search(z, x, loc, log_scale, weight, i, j):
    d = Normal(loc[:, i, j, :], log_scale[:, i, j, :].exp())
    w = weight[:, i, j, :]
    z1 = z[:, i, j]

    X = x.repeat((5, 1)).permute(1, 0)
    y = d.cdf(X)
    bl = (y * w).sum(dim=1)

    return z1, bl, (z1-bl).abs().max()


class AutFlow2DEstimator(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.losses = []
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        batch= dequantize(batch.float())
        _, _, _, loss = self.model(batch)
        self.losses.append(loss.detach().cpu().numpy().item())
        self.log('train/loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = dequantize(batch.float())
        _, _, _, loss = self.model(batch)
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
        # self.log('val/loss_after_epoch', loss, on_epoch=True, on_step=False)
        self.logger.log_metrics({'val/loss_after_epoch': loss},
                                step=self.trainer.current_epoch)


def pl_training_loop(train_data, test_data):
    global train_losses, test_losses, model, estimator, trainer

    batch_size = 64
    epochs = 1

    sz, H, W, C = train_data.shape

    train_ds = MnistDataset(train_data)
    test_ds = MnistDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = PixelCnnFlow(H, W, mixture_dim=5)
    estimator = AutFlow2DEstimator(model)
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


model = None
trainer = None
estimator = None
train_losses, test_losses = None, None
examples = None
train_d, test_d = None, None


def to_cuda(batch):
    if not CUDA:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def q2(train_data, test_data):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    H = W = 20
    Note that you should dequantize your train and test data, your dequantized pixels should all lie in [0,1]

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in [0, 1], where [0,0.5] represents a black pixel
        and [0.5,1] represents a white pixel. We will show your samples with and without noise.
    """

    global train_losses, test_losses, model, examples, train_d, test_d

    train_d = train_data
    test_d = test_data

    train_losses, test_losses, model = pl_training_loop(train_data, test_data)
    examples = model.generate_examples()

    return train_losses, test_losses, examples


if __name__ == '__main__':

    os.chdir('/home/ubik/projects/')
    seed_everything(1)
    q2_save_results(q2)

    # os.chdir('/home/ubik/projects/deepul/')
    # fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes.pkl'
    # H, W = 20, 20
    # train_data, test_data = load_pickled_data(fp)
    #
    # dset = 1
    #
    # model = PixelCnnFlow(H, W, debug=True)
    # examples = model.generate_examples()
    #
    # x = torch.ones((8, 1, H, W)).float()
    # x[1] = 0
    # y = model(x)
    #
    # examples = model.generate_examples()

    # n = Normal(dist.loc[0, 13, -1, 0].detach(), dist.scale[0, 13, -1, 0].detach())
    # orig[0, 13, -1, 0]
    # n.log_prob(0.5070).exp()
