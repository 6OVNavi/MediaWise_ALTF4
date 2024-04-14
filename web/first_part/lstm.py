import sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import random
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime, timezone





file_path = sys.argv[1]
user_folder = sys.argv[2]
timestamp_column = sys.argv[3]
header_row = sys.argv[4]
prediction_column = sys.argv[5]
length = sys.argv[6]
train = pd.read_excel(file_path)


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

    # PROCESSING

    num_cols = train.columns.values.tolist()
    num_cols.remove(timestamp_column)
    num_cols.remove(target_column)

    target_col = [target_column]
    train = train[num_cols + target_col]
    ss = StandardScaler()
    train[num_cols] = ss.fit_transform(train[num_cols])

    ss_target = StandardScaler()
    train[[target_column]] = ss_target.fit_transform(train[[target_column]])

    X_train, y_train = train[num_cols + target_col], train[target_col]

    num_preds = num_pred

    class TimeseriesDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, window=52, num_preds=29):
            self.X = X
            self.y = y
            self.window = window
            self.num_preds = num_preds

        def __len__(self):
            return self.X.__len__() - self.window + 1

        def __getitem__(self, index):
            return (self.X[index:index + self.window],
                    self.y[index + self.window:index + self.window + self.num_preds].reshape(-1))

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

    pred = ss_target.inverse_transform(pred.detach().numpy()).reshape(-1)

    return pred


class EXCELWorker:
    def __init__(self, xlsx_path, timestamp_column=None, column_string=None, na_border=50,
                 target_column="Продажи, рубли"):

        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.data = self.preprocess_init_data(xlsx_path, column_string, na_border)
        self.size = len(self.data)
        self.gbm_estimator = None
        self.train_part = None
        self.test_part = None
        self.test_dates = None

    def preprocess_init_data(self, xlsx_path, column_string, na_border):

        if column_string != None:
            df = pd.read_excel(xlsx_path, header=column_string)
        else:
            df = pd.read_excel(xlsx_path)

        df_nans_percent = df.isna().sum().values / len(df)
        na_border = df_nans_percent >= (na_border / 100)
        df = df.drop(columns=df.columns[na_border])

        return df

    def split(self, ):
        print(self.data)
        self.test_part = self.data[self.data[self.target_column].isna()]
        self.train_part = self.data[~self.data[self.target_column].isna()]

        if self.timestamp_column != None:
            self.train_part[self.timestamp_column] = self.train_part[self.timestamp_column].apply(
                lambda x: x.timestamp())

    def contain_nan(self, df):
        nans = self.data.isna().sum()
        return nans[nans != 0].to_dict()

    # available na strtagies = "mean", "max", "median", "min", "value", "backwardfill", "forwardfill"
    def preprocess_excel(self, na_strategies):

        for column in list(na_strategies.keys()):

            na_strategy = na_strategies[column]
            fill_value = 0

            if type(na_strategy) == int:
                fill_value = na_strategy
            if na_strategy == "max":
                fill_value = self.data[column].max()
            if na_strategy == "min":
                fill_value = self.data[column].min()
            if na_strategy == "mean":
                fill_value = self.data[column].mean()
            if na_strategy == "median":
                fill_value = self.data[column].median()
            if na_strategy == "forwardfill":
                self.data[column] = self.data[column].fillna(method="ffill")
            if na_strategy == "backwardfill":
                self.data[column] = self.data[column].fillna(method="backfill")
            else:
                self.data[column] = self.data[column].fillna(fill_value)

    def train_model_and_get_predict(self, num_preds=29):
        preds = get_predicts(self.train_part, self.timestamp_column, self.target_column, num_preds)
        return preds

    def init_gbm(self, verbose=0):
        X = self.train_part.drop(columns=[self.target_column])
        y = self.train_part.loc[:, self.target_column]
        gbm = CatBoostRegressor(iterations=200)
        gbm.fit(X, y, verbose=verbose)
        self.gbm_estimator = gbm

    def get_change_by_param(self, changes):

        if self.gbm_estimator != None:

            target_object = self.train_part.iloc[len(self.train_part) - 1, :]
            X = target_object.drop(columns=[self.target_column])
            y = target_object.loc[self.target_column]

            for column in list(changes.keys()):
                X[column] += X[column]

            result = self.gbm_estimator.predict([X])
            return (((result - y) / y) * 100)

        else:
            raise Exception("no fitted model")

    def get_feature_importances(self):

        X = self.train_part.drop(columns=[self.target_column])
        y = self.train_part.loc[:, self.target_column]
        explainer = shap.TreeExplainer(self.gbm_estimator)
        shap_values = explainer(X, y)

        values = sorted(self.gbm_estimator.feature_importances_, reverse=True)
        features_ids = sorted(range(len(X.columns)), reverse=True,
                              key=lambda x: self.gbm_estimator.feature_importances_[x])
        return shap_values, (values, features_ids)


worker = EXCELWorker(file_path, timestamp_column=timestamp_column, target_column=prediction_column,
                     column_string=int(header_row), na_border=100)
worker.split()
worker.train_part.fillna(0, inplace=True)
worker.train_part[worker.train_part.select_dtypes('object').columns] = worker.train_part.loc[:,
                                                                       worker.train_part.select_dtypes(
                                                                           'object').columns].replace(' ', 0).astype(
    'float')
# worker.init_gbm(verbose=0)
preds = worker.train_model_and_get_predict(num_preds=int(length))
timestamps = worker.test_part[timestamp_column]


pred_df = pd.DataFrame({'Timestamps': timestamps[:int(length)], 'Predicted_Revenue': preds})
pred_df.to_excel(file_path, index=False)

