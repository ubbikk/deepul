from torch.nn.modules.loss import CrossEntropyLoss
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


class Made(torch.nn.Module):
    def __init__(self, H, W, hidden_dim=None):
        super().__init__()
        self.H = H
        self.W = W
        self.d = H*W
        if hidden_dim is None:
            hidden_dim = self.d
        self.hidden_dim = hidden_dim

        self.h = torch.nn.Parameter(torch.zeros(self.d, self.hidden_dim))
        self.out = torch.nn.Parameter(torch.zeros(self.hidden_dim, self.d))
        self.h_bias = torch.nn.Parameter(torch.zeros(self.hidden_dim, ))
        self.out_bias = torch.nn.Parameter(torch.zeros(self.d, ))

        torch.nn.init.xavier_uniform(self.h.data)
        torch.nn.init.xavier_uniform(self.out.data)
        torch.nn.init.normal_(self.h_bias.data)
        torch.nn.init.normal_(self.out_bias.data)



    def forward(self, *input):
        x = input[0]
        x = x.float()


def q2_b(train_data, test_data, image_shape, dset_id):
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


if __name__ == '__main__':
    fp = '/home/ubik/projects/deepul/homeworks/hw1/data/hw1_data/shapes.pkl'
    train_data, test_data = load_pickled_data(fp)

    visualize_q2b_data(1)

    q2_save_results(1, 'b', q2_b)
