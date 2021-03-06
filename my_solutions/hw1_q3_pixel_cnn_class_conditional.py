import os

from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepul.hw1_helper import q3d_save_results
from deepul.utils import load_pickled_data

from deepul.hw1_helper import *

CUDA = torch.cuda.is_available()


class MnistDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.transpose(0, 3, 1, 2)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

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


class ClassConditionalMaskedConv2D(torch.nn.Conv2d):
    def __init__(self, mask, n_classes, in_channels, out_channels, kernel_size, stride=1,
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
        self.n_classes = n_classes
        conditional_bias = torch.zeros((n_classes, out_channels))
        torch.nn.init.xavier_uniform_(conditional_bias)
        self.conditional_bias = torch.nn.Parameter(conditional_bias, requires_grad=True)

    def forward(self, x, y) -> Tensor:
        res = F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        bias = self.conditional_bias[y]
        bias = bias.reshape(*bias.shape + (1, 1))
        res = res + bias
        return res


cache = {}


def get_hook(name):
    def hook(grad):
        cache[name] = grad

    return hook


class ClassConditionalPixelCNN(torch.nn.Module):
    def __init__(self, H, W, n_classes, debug=False):
        super().__init__()
        self.H = H
        self.W = W
        self.n_classes = n_classes
        self.debug = debug
        maskA = get_conv_mask(7, 'A')
        convA = ClassConditionalMaskedConv2D(maskA, self.n_classes,
                                             in_channels=1, out_channels=64,
                                             kernel_size=(7, 7), padding=3)
        maskB = get_conv_mask(7, 'B')
        convsB = [ClassConditionalMaskedConv2D(maskB, self.n_classes,
                                               in_channels=64, out_channels=64, kernel_size=(7, 7), padding=3) for i in
                  range(5)]
        conv1D1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        conv1D2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        convs = [convA] + convsB + [conv1D1, conv1D2]

        self.convs = torch.nn.ModuleList(convs)
        self.criterion = BCELoss(reduction='none')

    def forward(self, *input):
        x = input[0].float()
        y = input[1]
        b, c, H, W = x.shape
        # x = x.float()
        target = x.reshape(b * c * H * W).detach()
        probs = self.get_probs(x, y)

        loss = self.criterion(probs, target)

        return loss

    def get_probs(self, x, y):
        b, c, H, W = x.shape

        for i, conv in enumerate(self.convs[:-1]):
            if isinstance(conv, ClassConditionalMaskedConv2D):
                x = conv(x, y)
            else:
                x = conv(x)
            x = F.relu(x)
        last_conv = self.convs[-1]
        x = last_conv(x)

        x = x.reshape(b * c * H * W)
        probs = F.sigmoid(x)
        return probs

    def generate_examples(self, sz=100):
        self.eval()
        labels = torch.arange(self.n_classes).repeat(sz // self.n_classes).sort()[0]
        labels = to_cuda(labels)
        with torch.no_grad():
            inp = torch.zeros((sz, self.H, self.W), dtype=torch.float)
            inp = to_cuda(inp)
            self.pp = torch.zeros((sz, self.H, self.W), dtype=torch.float)

            for pos in range(self.H * self.W):
                probs = self.get_probs(inp.reshape(sz, 1, self.H, self.W), labels).reshape(sz, self.H, self.W).cpu()
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


def train_loop(model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=128):
    opt = Adam(model.parameters(), lr=1e-3)

    train_ds = MnistDataset(train_data, train_labels)
    test_ds = MnistDataset(test_data, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    losses = []
    test_losses = []
    for e in tqdm(range(epochs)):
        # for k, v in model.named_parameters():
        #     print(k, v.abs().mean())
        for b in train_loader:
            b = to_cuda(b)
            loss = model(*b).mean()

            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy().item())

            opt.zero_grad()

            # s = model.s.data.detach()
            # model.s = torch.nn.Parameter(s)

        with torch.no_grad():
            tmp = []
            for b in test_loader:
                b = to_cuda(b)
                loss = model(*b).mean()
                tmp.append(loss.cpu().numpy())

            test_losses.append(np.mean(tmp))
            print(test_losses[-1])

    return losses, test_losses, model.generate_examples()


model = None
losses, test_losses, distribution = None, None, None


def q3_d(train_data, train_labels, test_data, test_labels, image_shape, n_classes, dset_id):
    global model, losses, test_losses, examples
    """
    train_data: A (n_train, H, W, 1) numpy array of binary images with values in {0, 1}
    train_labels: A (n_train,) numpy array of class labels
    test_data: A (n_test, H, W, 1) numpy array of binary images with values in {0, 1}
    test_labels: A (n_test,) numpy array of class labels
    image_shape: (H, W), height and width
    n_classes: number of classes (4 or 10)
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets
  
    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, 1) of samples with values in {0, 1}
      where an even number of images of each class are sampled with 100 total
    """

    """ YOUR CODE HERE """

    H, W = image_shape
    print(f'CUDA is {CUDA}')
    model = ClassConditionalPixelCNN(H, W, n_classes)
    if CUDA:
        model.cuda()
    losses, test_losses, examples = train_loop(model, train_data, train_labels, test_data, test_labels, epochs=10,
                                               batch_size=128)

    return losses, test_losses, examples


if __name__ == '__main__':
    os.chdir('/home/ubik/projects/')

    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/shapes.pkl'
    H, W, n_classes = 20, 20, 4
    train_data, test_data, train_labels, test_labels = load_pickled_data(fp, include_labels=True)

    dset = 1

    train_ds = MnistDataset(train_data, train_labels)
    test_ds = MnistDataset(test_data, test_labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = ClassConditionalPixelCNN(H, W, n_classes, debug=False)
    samples = model.generate_examples(4)

    # q3d_save_results(1, q3_d)

    # fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/mnist.pkl'
    # H, W = 28, 28
    # train_data, test_data = load_pickled_data(fp)
    # dset = 2

    # q3a_save_results(1, q3_a)

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

    # H, W, n_classes = 20, 20, 4
    # model = ClassConditionalPixelCNN(H, W, n_classes, debug=True)

    # i = 131
    # sz = H * W
    # # x_np = (np.random.rand(H * W) > 0.5).astype(np.float)
    # # x = torch.from_numpy(x_np).float()
    # x = torch.ones((H, W)).float()
    # x.requires_grad = True
    # y = model(x.reshape((1, 1, H, W)))
    # loss = y[i]
    # loss.backward()
    #
    # max_dependent = torch.where((x.grad != 0))[0].max()
    # assert max_dependent < i
    #
    #
    # x = cache['inp'].squeeze()
    # bl = torch.nonzero(x.reshape(20**2))
