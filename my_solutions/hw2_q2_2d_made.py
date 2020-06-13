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
from deepul.hw1_helper import *
from torchviz import make_dot


# make_dot(r).render("attached", format="png")


class PairsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Made2DV0(torch.nn.Module):
    def __init__(self, d, emb_dim=10, hidden_dim=10):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb0 = torch.nn.Embedding(d, embedding_dim=emb_dim)
        self.emb1 = torch.nn.Embedding(d, embedding_dim=emb_dim)

        self.h0 = torch.nn.Linear(emb_dim, hidden_dim)
        self.h1 = torch.nn.Linear(emb_dim, hidden_dim)

        self.out1 = torch.nn.Linear(2 * hidden_dim, d)
        self.out2 = torch.nn.Linear(2 * hidden_dim, d)

    def forward(self, *input):
        x = input[0]

        x0 = x[:, 0]
        x1 = x[:, 1]

        x0 = self.emb0(x0)
        x1 = self.emb1(x1)

        h0 = self.h0(torch.zeros_like(x0))
        h0 = F.relu(h0)

        h1 = self.h1(x0)
        h1 = F.relu(h1)

        out0 = torch.cat([h0, torch.zeros_like(h1)], dim=-1)


class Made2D(torch.nn.Module):
    def __init__(self, d, emb_dim=10, hidden_dim=10):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb = torch.nn.Embedding(d, embedding_dim=emb_dim)

        self.h = torch.nn.Parameter(torch.zeros(hidden_dim, emb_dim))
        self.out0 = torch.nn.Parameter(torch.zeros(d, hidden_dim))
        self.out1 = torch.nn.Parameter(torch.zeros(d, hidden_dim))

        torch.nn.init.normal_(self.h.data)
        torch.nn.init.normal_(self.out0.data)
        torch.nn.init.normal_(self.out1.data)

        self.h_mask = torch.zeros((hidden_dim, emb_dim))
        self.out0_mask = torch.zeros((d, hidden_dim))
        # self.out1_mask = torch.zeros((d, hidden_dim))

        self.h_mask[hidden_dim / 2:, :emb_dim / 2] = 1
        self.out0_mask[:, :hidden_dim / 2] = 1

        self.criterion = CrossEntropyLoss()

    def forward(self, *input):
        x = input[0]
        target = input[0]

        x0 = x[:, 0]
        x1 = x[:, 1]

        x0 = self.emb(x0)
        x1 = self.emb(x1)

        x = torch.cat([x0, x1], dim=-1)
        x = ((self.h * self.h_mask) @ x.T).T
        x = F.relu(x)

        out0 = ((self.out0 * self.out0_mask) @ x.T).T
        out1 = (self.out1 @ x.T).T

        t0 = target[:, 0]
        t1 = target[:, 1]

        loss0 = self.criterion(out0, t0)
        loss1 = self.criterion(out1, t1)

        return (loss0 + loss1) / 2


def sample_data(dset_type):
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


def q2_a(train_data, test_data, d, dset_id):
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

    """ YOUR CODE HERE """


if __name__ == '__main__':
    # visualize_q2a_data(dset_type=1)

    batch_size = 64
    train_dist, test_dist, train_data, test_data = sample_data(1)
    train_ds = PairsDataset(train_data)
    test_ds = PairsDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    bsz = 4
    emb_d = 5
    h_d = 3
    h = torch.nn.Parameter(torch.zeros((h_d, emb_d)))
    torch.nn.init.xavier_uniform_(h.data)
    x = torch.rand((bsz, emb_d))

    res = (h @ x.T).T
