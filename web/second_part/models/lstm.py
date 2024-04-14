from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import os
import torch.nn as nn
import torch.optim as optim

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, window=52, num_preds=29):
        self.X = X
        self.y = y
        self.window = window
        self.num_preds = num_preds

    def __len__(self):
        return self.X.__len__() - self.window + 1

    def __getitem__(self, index):
        return (
        self.X[index:index + self.window], self.y[index + self.window:index + self.window + self.num_preds].reshape(-1))


def get_predicts(train, timestamp_column="timestamp", target_column="target", num_pred=29):
    def seed_everything(seed):  # Provide reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    timestamps = train[timestamp_column].values

    num_cols = train.columns.values.tolist()
    num_cols.remove(timestamp_column)
    num_cols.remove(target_column)

    target_col = [target_column]
    train = train[num_cols + target_col]

    # print(num_cols)
    ss = StandardScaler()
    train[num_cols] = ss.fit_transform(train[num_cols])

    ss_target = StandardScaler()
    train[[target_column]] = ss_target.fit_transform(train[[target_column]])

    X_train, y_train = train[num_cols + target_col], train[target_col]

    num_preds = num_pred

    data_length = train.shape[0]
    sliding_window = min(104, data_length // 2)

    train_dataset = TimeseriesDataset(torch.tensor(X_train.values), torch.tensor(y_train.values), window=sliding_window,
                                      num_preds=num_preds)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    num_features = X_train.shape[1]
    dataset_length = len(train_dataset)

    # MODEL

    class LstmTimeSeries(nn.Module):
        def __init__(
                self,
                hidden_size,
                num_layers,
                num_preds
        ):
            super(LstmTimeSeries, self).__init__()

            self.is_bidectional = False

            self.lstm = nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=self.is_bidectional
            )

            lstm_output_dim = hidden_size * (1 + self.is_bidectional)

            self.dropout1 = nn.Dropout(0.2)
            self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)

            self.dropout2 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 2)

            self.dropout3 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(lstm_output_dim // 2, num_preds)

            self.num_layers = num_layers
            self.hidden_size = hidden_size

        def forward(self, X, hidden=None, cell=None):
            out, (h_n, c_n) = self.lstm(X, (hidden, cell))
            out = torch.mean(out, dim=1)

            out = self.dropout1(out)
            out = self.fc1(out)
            out = self.dropout2(out)
            out = self.fc2(out)
            out = self.dropout3(out)
            out = self.fc3(out)

            return out, h_n, c_n

        def init_hidden(self, batch_size):
            h0 = torch.zeros(self.num_layers * (1 + self.is_bidectional), batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers * (1 + self.is_bidectional), batch_size, self.hidden_size).to(device)
            return h0, c0

    # Training Setup
    hidden_size = 128
    num_layers = 1
    learning_rate = 3e-2
    epochs = 15
    batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LstmTimeSeries(
        hidden_size,
        num_layers,
        num_preds
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = torch.nn.MSELoss(reduction='mean')

    # TRAINING

    seed_everything(42)

    for epoch in range(epochs):
        model.train()

        for count_batch, (X, y) in enumerate(tqdm(train_loader)):

            hidden, cell = model.init_hidden(batch_size=batch_size)

            if count_batch == dataset_length - 1: continue

            optimizer.zero_grad()

            pred, hidden, cell = model(X.float(), hidden, cell)

            loss = criterion(pred[:, :y.shape[1]], y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        loss = 0
        pred, _, _ = model(X.float(), hidden, cell)

    # print(pred)
    return model


train = pd.read_csv("/Users/kristianbogdan/PycharmProjects/Hack_UFO/gradio/data/train_sfo_processed_2.csv")

model = get_predicts(train, target_column="revenue")


def get_preds_koef(train, k, model, num_preds=29, target_column="revenue", timestamp_column="timestamp"):
    timestamps = train[timestamp_column].values

    num_cols = train.columns.values.tolist()
    num_cols.remove(timestamp_column)
    num_cols.remove(target_column)

    target_col = [target_column]
    train = train[num_cols + target_col]

    # print(num_cols)
    ss = StandardScaler()
    train[num_cols] = ss.fit_transform(train[num_cols])

    ss_target = StandardScaler()
    train[[target_column]] = ss_target.fit_transform(train[[target_column]])

    X_train, y_train = train[num_cols + target_col], train[target_col]

    batch_size = 1
    sliding_window = 104

    X_train.loc[185:214, 'disease_rate'] *= k
    train_dataset = TimeseriesDataset(torch.tensor(X_train.values), torch.tensor(y_train.values), window=sliding_window,
                                      num_preds=num_preds)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    hidden, cell = model.init_hidden(batch_size=batch_size)
    for count_batch, (X, y) in enumerate(tqdm(train_loader)):
        pass
    model.eval()
    pred, _, _ = model(X.float(), hidden, cell)
    pred = ss_target.inverse_transform(pred.detach().numpy()).reshape(-1)
    pred[pred < 0] = -pred[pred < 0] / 10
    return pred


# pred = get_preds_koef(train, 1, model=model, target_column="revenue")