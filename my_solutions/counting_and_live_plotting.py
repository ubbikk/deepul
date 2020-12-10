import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer, LightningModule, Callback
from torch.optim import Adam
import streamlit as st


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
        self.losses = []

    def training_step(self, batch, batch_idx):
        return self.model(*batch)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.model(*batch)
            self.log('val/loss', loss)
            return loss

    def validation_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean().numpy()
        self.losses.append(loss)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=3e-4)


class UpdatePlotCallback(Callback):
    def __init__(self, epochs, chart, progress_bar, status_text):
        self.epochs = epochs
        self.chart = chart
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_validation_epoch_end(self, trainer, pl_module):
        losses = pl_module.losses
        rows = np.array(losses[-1]).reshape(1,1)
        self.chart.add_rows(rows)

        epoch = trainer.current_epoch
        progress = int(((epoch+1)/self.epochs)*100)
        self.progress_bar.progress(progress)

        self.status_text.text(f'{progress}% Complete')

MAX_VAL = 100

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
status_text.text('0% Complete')
# last_rows = np.random.randn(1, 1)
chart = st.line_chart()
epochs = st.number_input("Enter a number", 1, 1000, 10)

dataset = CountDataset(max_val=MAX_VAL)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate)

model = CountModel(max_val=MAX_VAL)
estimator = CountEstimator(model)

plotting_callback = UpdatePlotCallback(epochs, chart, progress_bar, status_text)

trainer = Trainer(max_epochs=epochs,
                  callbacks=[plotting_callback],
                  check_val_every_n_epoch=1,
                  num_sanity_val_steps=0
                  )
trainer.fit(estimator, train_dataloader=loader, val_dataloaders=loader)

progress_bar.empty()
st.button("Re-run")