import os

from torch import Tensor
from torch.nn import BCELoss, CrossEntropyLoss, NLLLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepul.hw1_helper import q3a_save_results
from deepul.utils import load_pickled_data

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


class ResConvBlock(torch.nn.Module):
    def __init__(self, mask, in_channels, out_channels, h, kernel_size):
        super().__init__()
        self.conv = MaskedConv2D(mask,
                                 in_channels=h,
                                 out_channels=h,
                                 kernel_size=(kernel_size, kernel_size),
                                 padding=kernel_size // 2)

        self.in_conv = torch.nn.Conv2d(in_channels, h, kernel_size=1)
        self.out_conv = torch.nn.Conv2d(h, out_channels, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        inp = x
        x = self.in_conv(x)
        x = F.relu(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return x + inp


class PixelCNN(torch.nn.Module):
    def __init__(self, H, W, C, colors, num_filters=120):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.colors = colors
        self.num_filters = num_filters
        kernel_size = 7

        maskA = get_conv_mask(kernel_size, 'A')
        maskB = get_conv_mask(7, 'B')

        self.convA = MaskedConv2D(maskA,
                                  self.C,
                                  self.num_filters,
                                  kernel_size=(kernel_size, kernel_size),
                                  padding=kernel_size // 2)

        block0 = ResConvBlock(maskB,
                              in_channels=self.num_filters,
                              out_channels=num_filters,
                              h=num_filters,
                              kernel_size=kernel_size)

        blocks1_6 = [ResConvBlock(maskB,
                                  in_channels=num_filters,
                                  out_channels=num_filters,
                                  h=num_filters,
                                  kernel_size=kernel_size) for _ in range(6)]
        block7 = ResConvBlock(maskB,
                              in_channels=num_filters,
                              out_channels=num_filters,
                              h=num_filters,
                              kernel_size=kernel_size)

        self.out = torch.nn.Conv2d(in_channels=num_filters,
                                   out_channels=self.colors * self.C,
                                   kernel_size=1)

        blocks = [block0] + blocks1_6 + [block7]

        self.blocks = torch.nn.ModuleList(blocks)
        self.criterion = NLLLoss(reduction='none')

    def forward(self, *input):
        x = input[0]

        b, c, H, W = x.shape
        # x = x.float()
        target = x.reshape(b * c * H * W).detach()
        probs = self.get_log_probs(input).reshape(b * c * H * W, self.colors)

        loss = self.criterion(probs, target.long())

        return loss

    def get_log_probs(self, input):
        x = input[0]
        b, c, H, W = x.shape

        x = self.convA(x)
        x = F.relu(x)

        for block in self.blocks:
            # print(x.shape)
            x = block(x)
            x = F.relu(x)
            # print(x.shape)
            # print('========================')

        x = self.out(x)

        x = x.reshape(b, H, W, self.C, self.colors)
        probs = F.log_softmax(x, dim=-1)
        return probs

    def generate_examples(self, sz=100):
        self.eval()
        with torch.no_grad():
            inp = torch.zeros((sz, self.C, self.H, self.W), dtype=torch.float)
            inp = to_cuda(inp)

            for pos in range(self.H * self.W):
                scores = self.get_log_probs([inp]).cpu()  # .reshape(sz, self.H, self.W)
                probs = scores.exp()
                i = pos // self.H
                j = pos % self.H
                for b in range(sz):
                    for c in range(self.C):
                        p = probs[b, i, j, c].numpy()
                        inp[b, c, i, j] = np.random.choice(self.colors, p=p)

            return inp.reshape((sz, self.H, self.W, self.C)).cpu().numpy()


def to_cuda(batch):
    if not CUDA:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def train_loop(model, train_data, test_data, epochs=10, batch_size=128):
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


def q3_b(train_data, test_data, image_shape, dset_id):
    global model, losses, test_losses, examples
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """

    H, W, C = image_shape
    print(f'CUDA is {CUDA}')
    model = PixelCNN(H, W, C, 4)
    if CUDA:
        model.cuda()
    losses, test_losses, examples = train_loop(model, train_data, test_data, epochs=10, batch_size=128)

    return losses, test_losses, examples


if __name__ == '__main__':
    os.chdir('/home/ubik/projects/deepul/')

    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes_colored.pkl'
    H, W, C = 20, 20, 3
    train_data, test_data = load_pickled_data(fp)

    dset = 1

    # fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/mnist.pkl'
    # H, W = 28, 28
    # train_data, test_data = load_pickled_data(fp)
    # dset = 2

    model = PixelCNN(H, W, C, 4)
    examples = model.generate_examples(5)

    # q3bc_save_results(1, 'b', q3_b)

    # mask = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # conv = MaskedConv2D(mask, 5, 10, kernel_size=(3, 3), padding=1)
    #
    # inp = torch.rand((2, 5, 64, 64))
    #
    # out = conv(inp)
    # print(out.shape)
    #
    # conv1d = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)
    # out1 = conv1d(out)
    # print(out1.shape)

    # H, W = 20, 20
    # model = PixelCNN(H, W)
    #
    # i = 200
    # sz = H * W
    # x_np = (np.random.rand(H * W) > 0.5).astype(np.float)
    # x = torch.from_numpy(x_np).float()
    # x.requires_grad = True
    # y = model(x.reshape((1, 1, H, W)))
    # loss = y[i]
    # loss.backward()
    #
    # max_dependent = torch.where((x.grad != 0))[0].max()
    # assert max_dependent < i

    # residual blocks
    # Unknown dim of 1D conv
    # Layer Norm + autoregressive property? Needs some masking??
    # softmax instead of sigmoid:
    # additional dim for predictions
    # change sampling procedure to address softmax dim (argmax)
    # multi-channel?
    # multi-dumensional mask i,e 7x7x3
    # different output shape i.e. 3x more, additional dim

    # output dim: BxHxWx3x4

    # Where does additional dim(for softmax) come from?
    # BxHxWxC ==> BxHxWx12 ==> Reshape ==> BxHxWx3x4
