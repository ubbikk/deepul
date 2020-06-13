from collections import Counter

import torch
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

eps = 1e-10


# eps = torch.finfo(torch.float32).eps


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


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm


class MixtureOfLogisticsModel(torch.nn.Module):
    def __init__(self, d, mixture_size=4):
        super().__init__()
        self.d = d
        self.mixture_size = mixture_size
        self.pi = torch.nn.Parameter(torch.zeros(self.mixture_size))
        self.mu = torch.nn.Parameter(torch.zeros((self.mixture_size, 1)))
        self.s = torch.nn.Parameter(torch.zeros((self.mixture_size, 1)))

        # torch.nn.init.normal_(self.pi.data)
        # torch.nn.init.normal_(self.mu.data, self.d)
        # torch.nn.init.normal_(self.s.data, np.sqrt(self.d))

        torch.nn.init.ones_(self.s)
        torch.nn.init.uniform_(self.mu, 0, self.d)

        self.pis = []
        self.mus = []
        self.ss = []

    def get_log_probs_of_mixture(self):
        s = self.s ** 2
        # s = self.s

        positive = torch.arange(self.d).float() + 0.5
        positive[self.d - 1] = 10 ** 7
        positive = (positive - self.mu) / s
        positive = F.sigmoid(positive)

        negative = torch.arange(self.d).float() - 0.5
        negative[0] = -10 ** 7
        negative = (negative - self.mu) / s
        negative = F.sigmoid(negative)

        res = positive - negative

        res = res + eps
        # res = F.normalize(res + eps, p=1)
        # print(res.detach().numpy().max())
        # print(model.s.data.detach().flatten().numpy())

        pi = F.softmax(self.pi).reshape((self.mixture_size, 1))

        res = res * pi
        res = res.sum(dim=0)

        self.pis.append(self.pi.detach().numpy().copy())
        self.mus.append(self.mu.detach().numpy().copy())
        self.ss.append(self.s.detach().numpy().copy())

        return res.log()

    def get_mixture_params(self):
        with torch.no_grad():
            s = self.s ** 2
            mu = self.mu

            s = s.detach().squeeze().numpy()
            mu = mu.detach().squeeze().numpy()

            return list(zip(s, mu))

    def get__mixture_componets(self):
        with torch.no_grad():
            s = self.s ** 2
            # s = self.s

            positive = torch.arange(self.d).float() + 0.5
            positive[self.d - 1] = 10 ** 7
            positive = (positive - self.mu) / s
            positive = F.sigmoid(positive)

            negative = torch.arange(self.d).float() - 0.5
            negative[0] = -10 ** 7
            negative = (negative - self.mu) / s
            negative = F.sigmoid(negative)

            res = positive - negative

            return res.numpy()

    def get_mixture_coefficients(self):
        with torch.no_grad():
            pi = F.softmax(self.pi).reshape((self.mixture_size, 1))
            pi = pi.detach().squeeze()
            print(f'Mixture weights {pi}')

            return pi

    def forward(self, *input):
        sz = input[0].shape[0]
        scores = self.get_log_probs_of_mixture()
        scores = scores.repeat((sz, 1))

        return scores

    def get_distribution(self):
        theta = self.get_log_probs_of_mixture().detach().numpy()
        theta = np.exp(theta)
        return theta


def visualaize_components(model: MixtureOfLogisticsModel):
    thetas = model.get__mixture_componets()
    coefs = model.get_mixture_coefficients()
    params = model.get_mixture_params()

    for i in range(model.mixture_size):
        theta = thetas[i, :]
        c = coefs[i]
        s, mu = params[i]
        sample = sample_from_distribution(10000, theta)
        plt.figure()
        plt.title(f"Component #{i}, s={s:.3f}, mu={mu:.3f}, weight={c:.4f}, d={model.d}")
        plt.hist(sample, 100)


class HistDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def train_loop(model, train_data, test_data, epochs=10, batch_size=64):
    opt = Adam(model.parameters(), lr=1e-3)
    criterion = NLLLoss()

    train_ds = HistDataset(train_data)
    test_ds = HistDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    losses = []
    test_losses = []
    for e in tqdm(range(epochs)):
        for b in train_loader:
            scores = model(b)
            loss = criterion(scores, b)
            # loss += (-model.s.flatten()).relu().sum()
            loss.backward()
            # print(get_grad_norm(model))
            opt.step()
            losses.append(loss.detach().numpy().item())

            opt.zero_grad()

            # s = model.s.data.detach()
            # model.s = torch.nn.Parameter(s)

        with torch.no_grad():
            tmp = []
            for b in test_loader:
                scores = model(b)
                loss = criterion(scores, b)
                tmp.append(loss.numpy())

            test_losses.append(np.mean(tmp))
            print(test_losses[-1])

    return losses, test_losses, model.get_distribution()


def sample_from_distribution(sz, theta):
    return np.random.choice(range(len(theta)), sz, p=theta)


def visualize_params(pp):
    pp = np.stack(pp).squeeze()
    sz = pp.shape[-1]
    iterations = pp.shape[0]

    for i in range(sz):
        plt.plot(range(iterations), pp[:, i])

    plt.legend()


blja = None


def q1_b(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """
    global blja
    batch_size = 64
    mixture_size = 4
    model = MixtureOfLogisticsModel(d, mixture_size=mixture_size)
    losses, test_losses, theta = train_loop(model,
                                            train_data, test_data,
                                            epochs=200, batch_size=batch_size)

    blja = losses, test_losses, theta
    return losses, test_losses, theta


def theoretical_minimum_density(ww):
    ww = np.array(ww, dtype=np.float)
    ww += eps
    ww = ww / sum(ww)
    return -(ww * np.log(ww)).sum()


def theoretical_minimum_observations(ss):
    c = Counter(ss)
    ww = [v for v in c.values()]
    return theoretical_minimum_density(ww)


def artifitial_train_distrib1(sz=200):
    return [int(np.random.normal(200, 25)) for _ in range(sz)]


def artifitial_train_distrib2(sz=200):
    a = np.array([int(np.random.normal(30, 10)) for _ in range(sz // 2)])
    a = np.clip(a, 0, a.max())

    b = np.array([int(np.random.normal(80, 5)) for _ in range(sz // 2)])
    b = np.clip(b, 0, b.max())

    return np.concatenate([a, b]).tolist()


def cross_entropy(a, b):
    c1 = Counter(a)
    c2 = Counter(b)

    u = set(a) | set(b)

    mn = min(u)
    mx = max(u)

    w1 = [c1[i] for i in range(mn, mx + 1)]
    w2 = [c2[i] for i in range(mn, mx + 1)]

    w1 = np.array(w1, dtype=np.float)
    w2 = np.array(w2, dtype=np.float)

    w1 = w1 / w1.sum()
    w2 = w2 / w2.sum()

    res = sum([-x * np.log(y) for x, y in zip(w1, w2) if y != 0])

    return res


def quazy_logistic_density(s, mu, d):
    positive = torch.arange(d).float() + 0.5
    positive[d - 1] = 10 ** 7
    positive = (positive - mu) / s
    positive = F.sigmoid(positive)

    negative = torch.arange(d).float() - 0.5
    negative[0] = -10 ** 7
    negative = (negative - mu) / s
    negative = F.sigmoid(negative)

    res = positive - negative

    return res.numpy()


def sample_from_quazy_logistic(s, mu, d, sz=10_000):
    theta = quazy_logistic_density(s, mu, d)
    return sample_from_distribution(sz, theta)


def visualize_quazy_logistic(s, mu, d, sz=10_000):
    sample = sample_from_quazy_logistic(s, mu, d, sz=sz)
    plt.figure()
    plt.hist(sample, 100)
    plt.title(f'Hist for s={s:.3f},mu={mu:.3f},  d={d}')
    plt.show()


if __name__ == '__main__':
    seed_everything(0)

    distribution = artifitial_train_distrib2()
    real_distribution = artifitial_train_distrib2(10_000)

    train_data = np.array(distribution * 50)
    test_data = np.array(distribution * 10)

    # train_data = np.array(9000 * [5] + 1000 * [9])
    # test_data = np.array(910 * [5] + 90 * [9])
    #
    # train_data = np.array(6000 * [5] + 4000 * [9])
    # test_data = np.array(610 * [5] + 390 * [9])
    #
    # train_data = np.array(1000 * [1] + 5000 * [5] + 4000 * [9])
    # test_data = np.array(100 * [1] + 510 * [5] + 390 * [9])

    d = int(1.2 * max(train_data))

    mixture_size = 4
    model = MixtureOfLogisticsModel(d, mixture_size=mixture_size)

    # losses, test_losses, theta = q1_b(train_data, test_data, d, 1)
    losses, test_losses, theta = train_loop(model,
                                            train_data, test_data,
                                            epochs=200, batch_size=64)
    tt = model.get__mixture_componets()

    plt.figure()
    plt.title('Train loss')
    plt.plot(range(len(losses)), losses)

    plt.figure()
    plt.title('Test loss')
    plt.plot(range(len(test_losses)), test_losses)

    plt.figure()
    plt.hist(distribution, 50)
    plt.title('Train data')
    plt.show()

    plt.figure()
    plt.hist(real_distribution, 50)
    plt.title('Real distribution')
    plt.show()

    sample = sample_from_distribution(10000, theta)
    plt.figure()
    plt.title('Trained distribution')
    plt.hist(sample, 50)

    # plt.figure()
    # plt.title('pi')
    # visualize_params(model.pis)
    #
    # plt.figure()
    # plt.title('mu')
    # visualize_params(model.mus)
    #
    # plt.figure()
    # plt.title('s')
    # visualize_params(model.ss)

    visualaize_components(model)
