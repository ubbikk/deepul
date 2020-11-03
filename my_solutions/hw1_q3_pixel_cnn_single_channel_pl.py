from typing import Any, Union

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import DummyExperiment
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.nn import BCELoss
from torch.utils.data import Dataset, DataLoader

from deepul.hw1_helper import *

CUDA = torch.cuda.is_available()


class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data.transpose(0, 3, 1, 2)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def sample_from_bernulli_distr(p):
    return np.random.binomial(1, p, 1).item()


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

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class CustomLogger(LightningLoggerBase):

    def __init__(self, *args, **kwargs):
        super(CustomLogger, self).__init__()
        self._experiment = DummyExperiment()

    @property
    def experiment(self) -> Any:
        return self._experiment

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return 'blja'

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        return 1

    @rank_zero_only
    def log_hyperparams(self, params):
        print("--" * 30)
        print("this will never be printed")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print("--" * 30)
        print("the following will be an empty dict {}")
        print(metrics)


class PixelCnnEstimator(LightningModule):
    def __init__(self, H, W, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PixelCNN(H, W, debug=debug)
        self.losses = []
        self.test_losses = []

    def training_step(self, batch, batch_idx):
        preds = self.model(batch.float())
        loss = preds.mean()
        self.losses.append(loss.detach().cpu().numpy().item())
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('blja', torch.Tensor([2]), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch.float())
        loss = preds.mean()
        self.log('val/loss', loss, on_epoch=True, on_step=False)
        self.log('ff', torch.Tensor([2]))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def validation_epoch_end(self, outputs):
        ll = outputs['val/loss']
        loss = torch.mean(ll)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.test_losses.append(loss.detach().cpu().numpy().item())
        return loss


class PixelCNN(torch.nn.Module):
    def __init__(self, H, W, debug=False):
        super().__init__()
        self.H = H
        self.W = W
        self.debug = debug
        maskA = get_conv_mask(7, 'A')
        convA = MaskedConv2D(maskA, in_channels=1, out_channels=64,
                             kernel_size=(7, 7), padding=3)
        maskB = get_conv_mask(7, 'B')
        convsB = [MaskedConv2D(maskB, in_channels=64, out_channels=64, kernel_size=(7, 7), padding=3) for i in
                  range(5)]
        conv1D1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        conv1D2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        convs = [convA] + convsB + [conv1D1, conv1D2]

        self.convs = torch.nn.ModuleList(convs)
        self.criterion = BCELoss(reduction='none')

    def forward(self, *input):
        x = input[0]

        b, c, H, W = x.shape
        target = x.reshape(b * c * H * W).detach()
        probs = self.get_probs(input)

        loss = self.criterion(probs, target)

        return loss

    def get_probs(self, input):
        x = input[0]
        b, c, H, W = x.shape

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = F.relu(x)
        last_conv = self.convs[-1]
        x = last_conv(x)

        x = x.reshape(b * c * H * W)
        probs = torch.sigmoid(x)
        return probs

    def generate_examples(self, sz=100):
        self.eval()
        if CUDA:
            self.cuda()
        with torch.no_grad():
            inp = torch.zeros((sz, self.H, self.W), dtype=torch.float)
            inp = to_cuda(inp)
            self.pp = torch.zeros((sz, self.H, self.W), dtype=torch.float)

            for pos in range(self.H * self.W):
                probs = self.get_probs([inp.reshape(sz, 1, self.H, self.W)]).reshape(sz, self.H, self.W).cpu()
                i = pos // self.H
                j = pos % self.H
                for b in range(sz):
                    p = probs[b, i, j].numpy().item()
                    inp[b, i, j] = sample_from_bernulli_distr(p)
                    self.pp[b, i, j] = p

            return inp.reshape((sz, self.H, self.W, 1)).cpu().numpy()


def to_cuda(batch):
    if not CUDA:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def pl_training_loop(train_data, test_data, image_shape, dset_id):
    global estimator, model, trainer

    batch_size = 128
    epochs = 1

    train_ds = MnistDataset(train_data)
    test_ds = MnistDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    H, W = image_shape
    estimator = PixelCnnEstimator(H, W)
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

    losses = np.array(estimator.losses)
    test_losses = np.array(estimator.test_losses)

    samples = estimator.model.generate_examples()

    return losses, test_losses, samples


model = None
trainer = None
estimator = None
losses, test_losses, distribution = None, None, None


def q3_a(train_data, test_data, image_shape, dset_id):
    return pl_training_loop(train_data, test_data, image_shape, dset_id)


def check_autoregressive_property():
    H, W = 20, 20
    model = PixelCNN(H, W, debug=True)
    i = 131
    sz = H * W
    x = torch.ones((H, W)).float()
    x.requires_grad = True
    y = model(x.reshape((1, 1, H, W)))
    loss = y[i]
    loss.backward()
    max_dependent = torch.where((x.grad != 0))[0].max()
    assert max_dependent < i


if __name__ == '__main__':
    os.chdir('/home/ubik/projects/')

    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes.pkl'
    H, W = 20, 20
    train_data, test_data = load_pickled_data(fp)

    dset = 1

    q3a_save_results(1, q3_a)
