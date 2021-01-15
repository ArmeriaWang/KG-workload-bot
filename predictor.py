import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from utils import *

from clusterer import get_clusters
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_inout_sequences(input_data, tw):
    """
    切分数据
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class PredRNN(nn.Module):
    """
    LSTM模型
    """
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def predict_next_week_arrivals():
    """
    预测各列下一周到达率，存储在csv文件中
    """
    query_file, q_cluster = get_clusters(npass=200)

    # print(q_cluster)
    train_window = 7
    epochs = 100
    fut_pred = 7
    param_k = len(q_cluster)
    time_cnt = q_cluster[0].shape[1]
    print("param_k:", param_k)

    cluster_arrival = np.zeros((param_k, time_cnt))
    for i in range(param_k):
        cluster_arrival[i] = q_cluster[i].mean()
        plt.plot(range(time_cnt - fut_pred), cluster_arrival[i][:-fut_pred], label="cluster {}".format(i))

    plt.xlabel("Day")
    plt.ylabel("Cluster Arrival Rate")
    plt.title("Day vs Cluster Arrival Rate")
    plt.legend()
    plt.show()

    test_data_size = 7
    train_data = cluster_arrival[:, :-test_data_size]
    test_data = cluster_arrival[:, -test_data_size:]
    # print(train_data.shape[1])
    # print(test_data.shape[1])

    scaler = MinMaxScaler(feature_range=(-1, 1))

    all_cluster_prediction = []

    start_train_time = time.time()
    for cluster_id in range(param_k):
        train_data_normalized = scaler.fit_transform(train_data[cluster_id].reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

        use_gpu = torch.cuda.is_available()

        model = PredRNN()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if use_gpu:
            model = model.cuda()
            loss_function = loss_function.cuda()
        print(model)

        for i in range(epochs):
            single_loss = None
            for seq, labels in train_inout_seq:

                optimizer.zero_grad()

                if use_gpu:
                    seq, labels = seq.cuda(), labels.cuda()
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                                         torch.zeros(1, 1, model.hidden_layer_size).cuda())
                else:
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                         torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 25 == 1:
                if use_gpu:
                    single_loss = single_loss.cpu()
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        test_inputs = train_data_normalized[-train_window:].tolist()

        model.eval()

        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            if use_gpu:
                seq = seq.cuda()
            with torch.no_grad():
                if use_gpu:
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                                         torch.zeros(1, 1, model.hidden_layer_size).cuda())
                else:
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                         torch.zeros(1, 1, model.hidden_layer_size))
                test_inputs.append(model(seq).item())

        cur_cluster_actual_prediction = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

        all_cluster_prediction.append(cur_cluster_actual_prediction)

        # plt.title('Day vs Cluster {} Arrivals'.format(cluster_id))
        # plt.ylabel('Cluster Arrival')
        # plt.grid(True)
        # plt.autoscale(axis='x', tight=True)
        # plt.plot(cluster_arrival[cluster_id], label='actual arrivals')
        # plt.plot(range(time_cnt - train_window, time_cnt), cur_cluster_actual_prediction, label='predict arrivals')
        # plt.legend(loc='best')
        # plt.savefig('./pred_c{}.png'.format(cluster_id), bbox_inches='tight')
        # plt.show()

    print("train time:", time.time() - start_train_time)
    pred_result = pd.DataFrame(columns=range(fut_pred))
    err_mean = []
    err_var = []
    start_predict_time = time.time()
    for cluster_id in range(param_k):
        cluster_np = q_cluster[cluster_id].to_numpy()
        cluster_size = len(cluster_np)
        cluster_pred = all_cluster_prediction[cluster_id]
        for col_id in range(cluster_size):
            col_pre_arrival = cluster_np[col_id][:-test_data_size]
            mm_range = (np.min(col_pre_arrival), np.max(col_pre_arrival))
            name = q_cluster[cluster_id].index.values[col_id]
            col_pred = np.interp(cluster_pred, (cluster_pred.min(), cluster_pred.max()), mm_range)
            # col_pred = np.clip(all_cluster_prediction[cluster_id], np.min(col_pre_arrival), np.max(col_pre_arrival))
            new_df_row = col_pred.reshape(-1).tolist()
            pred_result.loc[name] = new_df_row

            # plt.title('Day vs Column {} Arrivals (cluster {})'.format(name, cluster_id))
            # plt.ylabel('Column Arrival')
            # plt.grid(True)
            # plt.autoscale(axis='x', tight=True)
            # plt.plot(range(time_cnt), q_cluster[cluster_id].iloc[col_id, :])
            # plt.plot(range(time_cnt - fut_pred, time_cnt), pred_result.loc[name])
            # plt.show()

            actual_ar = q_cluster[cluster_id].iloc[col_id, -fut_pred:].to_numpy() + 1
            print(actual_ar)
            pred_ar = pred_result.loc[name].to_numpy() + 1
            print(pred_ar)
            err = (pred_ar - actual_ar) / actual_ar
            err_mean.append(np.mean(err))
            err_var.append(np.var(err))
            print(np.mean(err), np.max(err))

        # break
    print("predict time:", time.time() - start_predict_time)
    print(pred_result)
    pred_file_name = './iofiles/pred_result_{}.csv'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    pred_result.to_csv(pred_file_name)

    print(err_mean)
    print(err_var)

    return query_file, pred_file_name


if __name__ == "__main__":
    predict_next_week_arrivals()
