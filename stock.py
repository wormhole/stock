import datetime

import numpy as np
import pandas as pd
import torch
import tushare as ts
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


def save_to_csv(code, start_date, end_date):
    """
    从tushare下载金融数据保存至csv
    :param code: 股票代码
    :param start_date: 开始时间
    :param end_date: 结束时间
    :return:
    """
    pro = ts.pro_api("16c1ecf7057f3358953105f9d23dbb156a01ac88cdb2ebca2917fd9e")
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df.to_csv(code + ".csv")


def read_from_csv(file, n=30, train_end=-500):
    """
    从csv读取股票数据
    :param file: 文件名
    :param n: 时间序列长度
    :param train_end: 划分训练集和测试集索引
    :return: df_trian, df_test, df, dates 训练集，测试集，所有数据，时间索引
    """
    df = pd.read_csv(file)
    df = df.sort_values("trade_date")
    df_trade_date = df["trade_date"]
    df = df[["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]]
    dates = []
    for date in df_trade_date.tolist():
        dates.append(datetime.datetime.strptime(str(date), "%Y%m%d"))
    df_train, df_test = df[:train_end], df[train_end - n:]
    return df_train, df_test, df, dates


def series_data(df, n):
    """
    将数据组装成时间序列
    :param df: DataFrame对像
    :param n: 时间序列长度
    :return: data, label 数据和标签的numpy数组对象
    """
    data = []
    label = []
    for i in range(len(df) - n):
        d = df[i:n + i].values
        l = df[n + i:n + i + 1][["open", "high", "low", "close"]].values
        data.append(d)
        label.append(l)
    return np.array(data), np.array(label)


def min_max_scaler(df):
    """
    数据归一化处理
    :param df: DataFrame对象
    :return: df,min_map,max_map 归一化后的DataFrame,每列的最小值，每列的最大值
    """
    df = df.copy()
    min_map = {}
    max_map = {}
    for idx in df:
        min = df[idx].min()
        max = df[idx].max()
        for i in df.index:
            df.loc[i, idx] -= min
            df.loc[i, idx] /= (max - min)
        min_map[idx] = min
        max_map[idx] = max
    return df, min_map, max_map


def standard_scaler(df):
    """
    数据标准化处理
    :param df: DataFrame对象
    :return: df,mean_map,std_map 标准化后的DataFrame对象，每列的均值，每列的标准差
    """
    df = df.copy()
    mean_map = {}
    std_map = {}
    for idx in df:
        mean = df[idx].mean()
        std = df[idx].std()
        for i in df.index:
            df.loc[i, idx] -= mean
            df.loc[i, idx] /= std
        mean_map[idx] = mean
        std_map[idx] = std
    return df, mean_map, std_map


def show(code, train_end, n):
    """
    画图显示，并预测下一个交易日的开盘价，最高价，收盘价，最低价
    :param code: 股票代码
    :param train_end: 训练集测试集分割下标
    :param n: 时间序列
    :return:
    """
    rnn = torch.load(code + ".pkl")
    device = torch.device("cuda:0")
    rnn.to(device)
    df_train, df_test, df, dates = read_from_csv(code + ".csv")

    # 进行验证并画图显示
    train = []
    test = []

    test_index = len(df) + train_end
    df_all_normal, mean, std = standard_scaler(df)
    means = np.array([mean["open"], mean["high"], mean["low"], mean["close"]])
    stds = np.array([std["open"], std["high"], std["low"], std["close"]])
    # df_all_normal, min, max = min_max_scaler(df)
    # mins = np.array([min["open"], min["high"], min["low"], min["close"]])
    # maxs = np.array([max["open"], max["high"], max["low"], max["close"]])
    ts_all_normal = torch.Tensor(df_all_normal.values)

    for i in range(n, len(df) + 1):
        x = ts_all_normal[i - n:i].to(device)
        x = torch.unsqueeze(x, dim=0)
        y = rnn(x).to(device)
        y = torch.squeeze(y).detach().cpu().numpy()[-1, :]
        yy = y * stds + means
        # yy = y * (maxs - mins) + mins
        if i < test_index:
            train.append(yy)
        elif (test_index <= i and i < len(df)):
            test.append(yy)
        else:
            print(yy)

    train = np.array(train)
    test = np.array(test)
    plt.rcParams["font.sans-serif"] = ["KaiTi"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(1)
    plt.subplot(221)
    plt.plot(dates[n:train_end], train[:, 0], color="#ff0000ff", label="训练集")
    plt.plot(dates[train_end:], test[:, 0], color="#0000ffff", label="测试集")
    plt.plot(dates, df["open"], color="#00ff00ff", label="真实数据", linestyle=":")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("开盘价")
    plt.legend()

    plt.subplot(222)
    plt.plot(dates[n:train_end], train[:, 1], color="#ff0000ff", label="训练集")
    plt.plot(dates[train_end:], test[:, 1], color="#0000ffff", label="测试集")
    plt.plot(dates, df["high"], color="#00ff00ff", label="真实数据", linestyle=":")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("最高价")
    plt.legend()

    plt.subplot(223)
    plt.plot(dates[n:train_end], train[:, 2], color="#ff0000ff", label="训练集")
    plt.plot(dates[train_end:], test[:, 2], color="#0000ffff", label="测试集")
    plt.plot(dates, df["low"], color="#00ff00ff", label="真实数据", linestyle=":")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("最低价")
    plt.legend()

    plt.subplot(224)
    plt.plot(dates[n:train_end], train[:, 3], color="#ff0000ff", label="训练集")
    plt.plot(dates[train_end:], test[:, 3], color="#0000ffff", label="测试集")
    plt.plot(dates, df["close"], color="#00ff00ff", label="真实数据", linestyle=":")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("收盘价")
    plt.legend()
    plt.show()


class RNN(torch.nn.Module):
    def __init__(self, input_size):
        """
        循环神经网络实现，采用LSTM
        :param input_size: 输入的特征维度
        """
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return:
        """
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data.float(), label.float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


# 超参数
# 学习率
LR = 0.0001
# EPOCH大小
EPOCH = 100
# 训练集和测试集划分索引
TRAIN_END = -500

if __name__ == "__main__":
    code = input("请输入股票代码:")
    N = int(input("请输入序列长度:"))
    # 从tushare下载股票历史日k数据
    save_to_csv(code, start_date="", end_date="")
    # 从csv中读取数据
    df_train, df_test, df, dates = read_from_csv(code + ".csv", N, TRAIN_END)
    # 将数据标准化或归一化处理
    df_train_normal, mean, std = standard_scaler(df_train)
    # df_train_normal, min, max = min_max_scaler(df_train)
    # 将数据组装成时间序列
    np_train_normal, np_label_normal = series_data(df_train_normal, N)
    # 将数据转成Tensor对象，并保存到DataLoader里
    ts_train_normal = torch.Tensor(np_train_normal)
    ts_label_normal = torch.Tensor(np_label_normal)
    train_set = TrainSet(ts_train_normal, ts_label_normal)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

    # 初始化神经网络，优化器，损失函数
    device = torch.device("cuda:0")
    rnn = RNN(9)
    rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    # 开始训练
    for step in range(EPOCH):
        for tx, ty in train_loader:
            # 将数据加载到GPU中
            tx = tx.to(device)
            ty = ty.to(device)
            output = rnn(tx)
            # 计算损失
            loss = loss_func(output[:, -1, :], torch.squeeze(ty))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
        # 打印每个epoch的损失
        print(step, loss)

    torch.save(rnn, code + ".pkl")
