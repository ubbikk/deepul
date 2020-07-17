import os

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepul.hw1_helper import q3a_save_results
from deepul.utils import load_pickled_data

from deepul.hw1_helper import *

CUDA = torch.cuda.is_available()


class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data.squeeze()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


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

        self.register_buffer('mask', mask.reshape(1, in_channels, *kernel_size))

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PixelCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H
        self.W = W
        self.d = H * W

    def forward(self, *input):
        pass


def to_cuda(batch):
    if not CUDA:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def train_loop(model, train_data, test_data, epochs=100, batch_size=64):
    opt = Adam(model.parameters(), lr=1e-3)

    train_ds = MnistDataset(train_data)
    test_ds = MnistDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    losses = []
    test_losses = []
    for e in tqdm(range(epochs)):
        # for k, v in model.named_parameters():
        #     print(k, v.abs().mean())
        for b in train_loader:
            b = to_cuda(b).float()
            loss = model(b).mean()

            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy().item())

            opt.zero_grad()

            # s = model.s.data.detach()
            # model.s = torch.nn.Parameter(s)

        with torch.no_grad():
            tmp = []
            for b in test_loader:
                b = to_cuda(b).float()
                loss = model(b).mean()
                tmp.append(loss.cpu().numpy())

            test_losses.append(np.mean(tmp))
            print(test_losses[-1])

    return losses, test_losses, model.generate_examples()


model = None
losses, test_losses, distribution = None, None, None


def q3_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    H, W = image_shape
    print(f'CUDA is {CUDA}')
    model = PixelCNN(H, W)
    if CUDA:
        model.cuda()
    losses, test_losses, examples = train_loop(model, train_data, test_data, epochs=20, batch_size=64)

    return losses, test_losses, examples


if __name__ == '__main__':
    os.chdir('/home/ubik/projects/deepul/')

    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes.pkl'
    H, W = 20, 20
    train_data, test_data = load_pickled_data(fp)
    dset = 1

    # fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/mnist.pkl'
    # H, W = 28, 28
    # train_data, test_data = load_pickled_data(fp)
    # dset = 2

    # q3a_save_results(2, q3_a)

    mask = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    l = MaskedConv2D(mask, 1, 1, (3,3))

    inp = torch.rand((2,3,3))

    out = l(inp)