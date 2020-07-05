from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from deepul.hw1_helper import *

CUDA = torch.cuda.is_available()


class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data.squeeze()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def sample_from_bernulli_distr(p):
    return np.random.binomial(1, p, 1).item()


class MaskedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)


class Made(torch.nn.Module):
    def __init__(self, H, W, hidden_dim=None):
        super().__init__()
        self.H = H
        self.W = W
        self.d = H * W
        if hidden_dim is None:
            hidden_dim = self.d
        self.hidden_dim = hidden_dim

        # self.h = torch.nn.Parameter(torch.zeros(self.d, self.hidden_dim))
        # self.out = torch.nn.Parameter(torch.zeros(self.hidden_dim, self.d))
        # self.h_bias = torch.nn.Parameter(torch.zeros(self.hidden_dim, ))
        # self.out_bias = torch.nn.Parameter(torch.zeros(self.d, ))
        #
        # torch.nn.init.xavier_uniform(self.h.data)
        # torch.nn.init.xavier_uniform(self.out.data)
        # torch.nn.init.normal_(self.h_bias.data)
        # torch.nn.init.normal_(self.out_bias.data)

        h_mask = torch.ones(self.d, self.hidden_dim)
        h_mask = torch.tril(h_mask, -1)
        self.h = MaskedLinear(self.d, self.hidden_dim, h_mask)

        out_mask = torch.ones(self.hidden_dim, self.d)
        out_mask = torch.tril(out_mask)
        self.out = MaskedLinear(self.hidden_dim, self.d, out_mask)
        self.criterion = BCELoss(reduction='none')  # reduction='sum'

    def forward(self, *input):
        x = input[0]

        b, H, W = x.shape
        # x = x.float()
        target = x.reshape(b * H * W).detach()

        probs = self.get_probs(input)
        loss = self.criterion(probs, target)

        return loss

    def get_probs(self, input):
        x = input[0]
        inp = input[0]

        b, H, W = x.shape
        # target = x.reshape(b * H * W)
        x = x.reshape((b, H * W))
        # x = x.float()
        x = self.h(x)
        x = F.relu(x)
        x = self.out(x)
        probs = F.sigmoid(x)
        probs = probs.reshape(b * H * W)
        return probs

    def generate_examples(self, sz=100):
        self.eval()
        with torch.no_grad():
            inp = torch.zeros((sz, self.H, self.W), dtype=torch.float)
            inp = to_cuda(inp)
            self.pp = torch.zeros((sz, self.H, self.W), dtype=torch.float)

            for pos in range(self.H * self.W):
                probs = self.get_probs([inp]).reshape(sz, self.H, self.W).cpu()
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


def q2_b(train_data, test_data, image_shape, dset_id):
    global model, losses, test_losses, examples
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    """ YOUR CODE HERE """
    H, W = image_shape
    print(f'CUDA is {CUDA}')
    model = Made(H, W)
    if CUDA:
        model.cuda()
    losses, test_losses, examples = train_loop(model, train_data, test_data, epochs=20, batch_size=64)

    return losses, test_losses, examples


def test_autoregressive_property():
    sz = H * W
    x_np = (np.random.rand(H * W) > 0.5).astype(np.float)

    for i in range(sz):
        x = torch.from_numpy(x_np).float()
        x.requires_grad = True
        y = model(x.reshape((1, H, W)))
        loss = y[i]
        loss.backward()


if __name__ == '__main__':
    os.chdir('/home/ubik/projects/deepul/')

    # fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes.pkl'
    # H, W = 20, 20
    # train_data, test_data = load_pickled_data(fp)
    # dset = 1

    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/mnist.pkl'
    H, W = 28, 28
    train_data, test_data = load_pickled_data(fp)
    dset = 2

    # q2_save_results(1, 'b', q2_b)
    model = Made(H, W)
    q2_save_results(dset, 'b', q2_b)
    # visualize_q2b_data(dset)
    # show_samples(255*examples, title=f'Generated examples')

    i = 100
    sz = H * W
    x_np = (np.random.rand(H * W) > 0.5).astype(np.float)
    x = torch.from_numpy(x_np).float()
    x.requires_grad = True
    y = model(x.reshape((1, H, W)))
    loss = y[i]
    loss.backward()

    max_dependent = torch.where((x.grad != 0))[0].max()
    assert max_dependent < i

    # 0.0623, 0.1106
