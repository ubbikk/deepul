import torch
from pytorch_lightning import LightningModule, Trainer
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset, DataLoader

from deepul.hw2_helper import *

CUDA = torch.cuda.is_available()
torch.autograd.detect_anomaly()


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


def q3_a(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

    """ YOUR CODE HERE """

def visualize(train_data, sz=1):
    idxs = np.random.choice(len(train_data), replace=False, size=(sz,))
    images = train_data[idxs].astype(np.float32) / 3.0 * 255.0
    samples = (torch.FloatTensor(images) / 255).permute(0, 3, 1, 2)
    plt.imshow(samples[0].permute(1, 2, 0))

if __name__ == '__main__':
    os.chdir('/home/ubik/projects/')
    seed_everything(1)
    # q3_save_results(q3_a, 'a')
    train_data, test_data = load_pickled_data('deepul/homeworks/hw2/data/celeb.pkl')