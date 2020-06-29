from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepul.hw1_helper import *


class MnistDataset(Dataset):
    def __init__(self, data):
        self.data = data.squeeze()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def sample_from_bernulli_distr(p):
    return np.random.binomial(1, p, 1).item()


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

        self.h = torch.nn.Linear(self.d, self.hidden_dim)
        self.h.weight.data = torch.tril(self.h.weight.data, -1)

        self.out = torch.nn.Linear(self.hidden_dim, self.d)
        self.out.weight.data = torch.tril(self.out.weight.data)
        self.criterion = BCELoss()

    def forward(self, *input):
        x = input[0]

        b, H, W = x.shape
        x = x.float()
        target = x.reshape(b * H * W)

        probs = self.get_probs(input)
        loss = self.criterion(probs, target)

        return loss

    def get_probs(self, input):
        x = input[0]
        b, H, W = x.shape
        # target = x.reshape(b * H * W)
        x = x.reshape((b, H * W))
        x = x.float()
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

            for pos in range(self.H * self.W):
                probs = self.get_probs([inp]).reshape(sz, self.H, self.W)
                i = pos // self.H
                j = pos % self.H
                for b in range(sz):
                    inp[b, i, j]= sample_from_bernulli_distr(probs[b, i, j].numpy().item())

            return inp.reshape((sz, self.H, self.W, 1))


def to_cuda(batch, cuda):
    if not cuda:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def train_loop(model, train_data, test_data, cuda, epochs=100, batch_size=64):
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
            b = to_cuda(b, cuda)
            loss = model(b)

            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy().item())

            opt.zero_grad()

            # s = model.s.data.detach()
            # model.s = torch.nn.Parameter(s)

        with torch.no_grad():
            tmp = []
            for b in test_loader:
                b = to_cuda(b, cuda)
                loss = model(b)
                tmp.append(loss.cpu().numpy())

            test_losses.append(np.mean(tmp))
            print(test_losses[-1])

    return losses, test_losses, model.generate_examples()


model = None
losses, test_losses, distribution = None, None, None


def q2_b(train_data, test_data, image_shape, dset_id):
    global model, losses, test_losses, distribution
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
    cuda = torch.cuda.is_available()
    print(f'CUDA is {cuda}')
    model = Made(H, W)
    if cuda:
        model.cuda()
    losses, test_losses, examples = train_loop(model, train_data, test_data, cuda, epochs=5, batch_size=64)

    return losses, test_losses, examples


if __name__ == '__main__':
    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes.pkl'
    train_data, test_data = load_pickled_data(fp)



    # q2_save_results(1, 'b', q2_b)
    H, W = 20, 20
    model = Made(H, W)
    losses, test_losses, examples = train_loop(model, train_data, test_data, False, epochs=100, batch_size=64)
    visualize_q2b_data(1)
    show_samples(255*examples, title=f'Generated examples')
