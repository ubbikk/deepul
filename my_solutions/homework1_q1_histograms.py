import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class HistModel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.theta = torch.nn.Parameter(torch.FloatTensor(d))
        torch.nn.init.zeros_(self.theta.data)

    def forward(self, *input):
        sz = input[0].shape[0]
        scores = self.theta.repeat((sz, 1))

        return scores

    def get_theta(self):
        theta = F.softmax(self.theta.data).numpy()
        return theta


class HistDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def train_loop(model, train_data, test_data, epochs=10, batch_size=4):
    opt = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

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
            loss.backward()
            opt.step()
            losses.append(loss.detach().numpy().item())

            opt.zero_grad()

        with torch.no_grad():
            tmp = []
            for b in test_loader:
                scores = model(b)
                loss = criterion(scores, b)
                tmp.append(loss.numpy())

            test_losses.append(np.mean(tmp))

    return losses, test_losses, model.get_distribution()


def sample_from_distribution(d, sz, theta):
    return np.random.choice(range(d), sz, p=theta)


def q1_a(train_data, test_data, d, dset_id):
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

    batch_size = 64
    model = HistModel(d)
    losses, test_losses, theta = train_loop(model,
                                            train_data, test_data,
                                            epochs=10, batch_size=batch_size)

    return losses, test_losses, theta


if __name__ == '__main__':
    d = 10
    model = HistModel(10)

    train_data = np.array(9000 * [5] + 1000 * [9])
    test_data = np.array(910 * [5] + 90 * [9])

    losses, test_losses, theta = q1_a(train_data, test_data, d, 1)

    plt.figure()
    plt.plot(range(len(losses)), losses)

    plt.figure()
    plt.plot(range(len(test_losses)), test_losses)

    sample = sample_from_distribution(d, 1000, theta)
    plt.figure()
    plt.hist(sample)
