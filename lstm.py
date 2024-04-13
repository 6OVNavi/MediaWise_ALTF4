import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_percentage_error as mape, r2_score
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import os
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from datetime import datetime


def seed_everything(seed):    # Provide reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
os.makedirs('checkpoints', exist_ok=True)


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

timestamps = train['week_beg'].values*100
test_timestamps = [timestamps[-1] + 604800 * i for i in range(1, 30)]

train[train.select_dtypes('object').columns] = train.loc[:, train.select_dtypes('object').columns].replace(' ', 0).astype('float')
test[test.select_dtypes('object').columns] = test.loc[:, test.select_dtypes('object').columns].replace(' ', 0).astype('float')

# Processing

train.insert(0, 'week_number', train.index.values)

for col in ['_competitor_rating',
 '_competitor_coverage',
 '_competitor_money',
 '_competitor_digital',
 '_competitor_sponsorship_money',
 '_competitor_oon_money',
 '_competitor_radio_money',
 '_competitor_total']:

    train['competitors_'+col.rsplit('_', 1)[-1]] = train[[str(i) + col for i in range(1, 14)]].sum(axis=1)
    train.drop(columns=[str(i) + col for i in range(1, 14)], inplace=True)


col = '_competitor_tv_reg'
train['competitors_tv_reg'] = train[[str(i) + '_competitor_tv_reg' for i in [3, 5, 6, 7, 9, 12]]].sum(axis=1)
train.drop(columns=[str(i) + '_competitor_tv_reg' for i in [3, 5, 6, 7, 9, 12]], inplace=True)

for col in ['_video_rating',
 '_video_money',
 '_video_coverage_5']:

    train.drop(columns=[str(i) + col for i in range(1, 5)], inplace=True)


num_cols = train.columns.values.tolist()
num_cols.remove('week_beg')
num_cols.remove('revenue')

target_col = ['revenue']
train = train[num_cols+target_col]

ss = StandardScaler()
train[num_cols] = ss.fit_transform(train[num_cols])

ss_target = StandardScaler()
train[['revenue']] = ss_target.fit_transform(train[['revenue']])



X_train, y_train = train[num_cols+target_col], train[target_col]

num_preds = 29

y_train_unscaled = ss_target.inverse_transform(y_train)


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, window=52, num_preds=29):
        self.X = X
        self.y = y
        self.window = window
        self.num_preds = num_preds

    def __len__(self):
        return self.X.__len__() - self.window + 1

    def __getitem__(self, index):
        return (self.X[index:index+self.window], self.y[index+self.window:index+self.window+self.num_preds].reshape(-1))

    
    
sliding_window = 104

train_dataset = TimeseriesDataset(torch.tensor(X_train.values), torch.tensor(y_train.values), window=sliding_window, num_preds=num_preds)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False)

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
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim//2)

        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(lstm_output_dim//2, lstm_output_dim//2)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(lstm_output_dim//2, num_preds)

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
learning_rate = 4e-4
epochs = 10
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
        # if y.shape[1] < 29: continue

        hidden, cell = model.init_hidden(batch_size=batch_size)

        if count_batch == dataset_length - 1: continue

        optimizer.zero_grad()

        pred, hidden, cell = model(X.float(), hidden, cell)

        loss = criterion(pred[:, :y.shape[1]], y.float())
        loss.backward()
        optimizer.step()

    # print(f'Epoch: {epoch}, Train RMSE: {round(loss.item()**0.5)}')

    model.eval()
    loss = 0
    pred, _, _ = model(X.float(), hidden, cell)
    
pred = ss_target.inverse_transform(pred.detach().numpy()).reshape(-1)
train_revenue = ss_target.inverse_transform(train[['revenue']].values).reshape(-1)


weeks = [datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d') for timestamp in timestamps]
test_weeks = [datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d') for timestamp in test_timestamps]

plt.figure(figsize=(13,10))
plt.plot(weeks, train_revenue/10**6, label='Historical revenue')
plt.plot(test_weeks, pred/10**6, label='Predicted revenue')
plt.title('Историческая и предсказанная выручка') 
plt.xlabel('Дата')
plt.ylabel('Выручка, миллионов')
plt.legend()
plt.xticks((weeks+test_weeks)[::10] , fontsize=9, rotation=45)
plt.show()