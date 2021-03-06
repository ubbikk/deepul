import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer, LightningModule
from torch.optim import Adam


class CountDataset(Dataset):
    def __init__(self, sz=10_000, max_val=100, max_len=16):
        self.sz = sz
        self.max_val = max_val
        self.max_len = max_len
        self.sizes = 1 + torch.randint(max_len - 1, (sz,))

    def __getitem__(self, item):
        s = self.sizes[item]
        data = torch.randint(self.max_val - 1, (s,))
        return data, data.sum()

    def __len__(self):
        return self.sz


def collate(data):
    dd = [d[0] for d in data]
    dd = pad_sequence(dd, batch_first=True)

    target = [d[1] for d in data]
    target = torch.stack(target)

    return dd, target


class CountModel(torch.nn.Module):
    def __init__(self, max_val=100):
        super().__init__()
        self.max_val = max_val
        self.embeddings = torch.nn.Embedding(num_embeddings=self.max_val, embedding_dim=50)
        self.rnn = torch.nn.GRU(input_size=50, hidden_size=64, batch_first=True)
        self.fc = torch.nn.Linear(in_features=64, out_features=1)
        self.criterion = torch.nn.MSELoss()

    def forward(self, *input):
        x = input[0]
        y = input[1]

        x = self.embeddings(x)
        x, _ = self.rnn(x)
        x = self.fc(x)

        loss = self.criterion(x, y.float())

        return loss


class CountEstimator(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def training_step(self, batch, batch_idx):
        return self.model(*batch)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.model(*batch)
            self.log('val/loss', loss)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=3e-4)


MAX_VAL = 100

if __name__ == '__main__':
    dataset = CountDataset(max_val=MAX_VAL)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate)

    model = CountModel(max_val=MAX_VAL)
    estimator = CountEstimator(model)

    # ll = list(loader)
    # batch = ll[0]
    #
    # out = model(*batch)

    trainer = Trainer(max_epochs=10,
                      check_val_every_n_epoch=1,
                      # num_sanity_val_steps=0
                      )
    trainer.fit(estimator, train_dataloader=loader, val_dataloaders=loader)