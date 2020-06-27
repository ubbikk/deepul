from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepul.hw1_helper import *


# make_dot(r).render("attached", format="png")


class PairsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Made2D(torch.nn.Module):
    def __init__(self, d, emb_dim=25, hidden_dim=50):
        super().__init__()
        self.d = d
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb1 = torch.nn.Embedding(d, embedding_dim=emb_dim)
        self.emb2 = torch.nn.Embedding(d, embedding_dim=emb_dim)

        self.h = torch.nn.Parameter(torch.zeros(hidden_dim, 2 * emb_dim))
        self.out0 = torch.nn.Parameter(torch.zeros(d, hidden_dim))
        self.out0_bias = torch.nn.Parameter(torch.zeros(d, ))
        self.out1 = torch.nn.Parameter(torch.zeros(d, hidden_dim))
        self.out1_bias = torch.nn.Parameter(torch.zeros(d, ))

        torch.nn.init.normal_(self.h.data)
        torch.nn.init.normal_(self.out0.data)
        torch.nn.init.normal_(self.out1.data)
        torch.nn.init.normal_(self.out0_bias.data)
        torch.nn.init.normal_(self.out1_bias.data)

        self.h_mask = torch.zeros((hidden_dim, 2 * emb_dim))
        self.out0_mask = torch.zeros((d, hidden_dim))
        # self.out1_mask = torch.zeros((d, hidden_dim))
        if torch.cuda.is_available():
            self.h_mask = self.h_mask.cuda()
            self.out0_mask = self.out0_mask.cuda()

        self.h_mask[hidden_dim // 2:, :emb_dim] = 1
        self.out0_mask[:, :hidden_dim // 2] = 1

        self.criterion = CrossEntropyLoss()

    def get_distribution(self):
        self.eval()
        with torch.no_grad():
            inp = torch.cartesian_prod(torch.arange(self.d), torch.arange(self.d))
            if torch.cuda.is_available():
                inp = inp.cuda()
            s0, s10 = self.get_scores([inp])

            p0 = F.softmax(s0, dim=1)
            p10 = F.softmax(s10, dim=1)

            i0 = inp[:, 0]
            i1 = inp[:, 1]
            pt0 = p0[torch.arange(p0.shape[0]), i0]
            pt10 = p10[torch.arange(p10.shape[0]), i1]

            res = pt0 * pt10
            res = res.reshape((self.d, self.d))

            return res.cpu().numpy()

    def forward(self, *input):
        target = input[0]
        t0 = target[:, 0]
        t1 = target[:, 1]

        out0, out1 = self.get_scores(input)

        loss0 = self.criterion(out0, t0)
        loss1 = self.criterion(out1, t1)

        return (loss0 + loss1) / 2

    def get_scores(self, input):
        x = input[0]
        x0 = x[:, 0]
        x1 = x[:, 1]
        x0 = self.emb1(x0)
        x1 = self.emb2(x1)
        x = torch.cat([x0, x1], dim=-1)
        x = ((self.h * self.h_mask) @ x.T).T
        x = F.relu(x)

        out0 = (self.out0 * self.out0_mask)
        out0 = (out0 @ x.T).T
        out0 = out0 + self.out0_bias

        out1 = (self.out1 @ x.T).T
        out1 = out1 + self.out1_bias

        return out0, out1


def sample_data_copy(dset_type):
    data_dir = get_data_dir(1)
    # n=dataset sise (80/20)
    # d = dimension i.e. picture is d x d, so emb matrices are going be d x E
    if dset_type == 1:
        n, d = 10000, 25
        true_dist, data = q2_a_sample_data(join(data_dir, 'smiley.jpg'), n, d)
    elif dset_type == 2:
        n, d = 100000, 200
        true_dist, data = q2_a_sample_data(join(data_dir, 'geoffrey-hinton.jpg'), n, d)
    else:
        raise Exception('Invalid dset_type:', dset_type)
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]

    train_dist, test_dist = np.zeros((d, d)), np.zeros((d, d))
    for i in range(len(train_data)):
        train_dist[train_data[i][0], train_data[i][1]] += 1
    train_dist /= train_dist.sum()

    for i in range(len(test_data)):
        test_dist[test_data[i][0], test_data[i][1]] += 1
    test_dist /= test_dist.sum()

    return train_dist, test_dist, train_data, test_data


def to_cuda(batch, cuda):
    if not cuda:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.cuda()

    return [b.cuda() for b in batch]


def train_loop(model, train_data, test_data, cuda, epochs=100, batch_size=64):
    opt = Adam(model.parameters(), lr=1e-3)

    train_ds = PairsDataset(train_data)
    test_ds = PairsDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    losses = []
    test_losses = []
    for e in tqdm(range(epochs)):
        for k, v in model.named_parameters():
            print(k, v.abs().max())
        for b in train_loader:
            b = to_cuda(b, cuda)
            loss = model(b)
            # loss += (-model.s.flatten()).relu().sum()
            loss.backward()
            # print(get_grad_norm(model))
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

    return losses, test_losses, model.get_distribution()

model = None
losses, test_losses, distribution = None, None, None


def q2_a(train_data, test_data, d, dset_id):
    global model, losses, test_losses, distribution
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """
    cuda = torch.cuda.is_available()
    print(f'CUDA is {cuda}')
    model = Made2D(d)
    if cuda:
        model.cuda()
    losses, test_losses, distribution = train_loop(model, train_data, test_data, cuda, epochs=10, batch_size=512)

    return losses, test_losses, distribution


if __name__ == '__main__':
    # visualize_q2a_data(dset_type=1)

    dset=1
    d = 25
    batch_size = 64
    train_dist, test_dist, train_data, test_data = sample_data_copy(dset)

    # dset=2
    # d = 200
    # batch_size = 64
    # train_dist, test_dist, train_data, test_data = sample_data_copy(dset)

    # train_ds = PairsDataset(train_data)
    # test_ds = PairsDataset(test_data)
    #
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    #
    # bsz = 4
    # emb_d = 5
    # h_d = 3
    # h = torch.nn.Parameter(torch.zeros((h_d, emb_d)))
    # torch.nn.init.xavier_uniform_(h.data)
    # x = torch.rand((bsz, emb_d))
    #
    # res = (h @ x.T).T

    # model = Made2D(d)
    #
    # losses, test_losses, distribution = train_loop(model, train_data, test_data, False, epochs=50)
    # q2_a(train_data, test_data, d, 1)
    q2_save_results(dset, 'a', q2_a)
    visualize_q2a_data(dset)

    # 3.1980457 3.1819618
    # 5.294696