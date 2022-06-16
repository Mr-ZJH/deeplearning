import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import torch.nn as nn


# 构建网络结构
class Net(nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能

        self.classifier = nn.Sequential(
            nn.Linear(n_feature, 300),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(300, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, n_output))

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        x = self.classifier(x)
        return x


"""
数据集均放在csv文件中，分为train.csv和test.csv
从column_name1到column_name2是features
从column_name3到column_name4是labels
"""


def label_mean_std():
    o_train = pd.read_csv('your path/train.csv', encoding='utf-8')
    o_test = pd.read_csv('your path/test.csv', encoding='utf-8')
    all_labels = pd.concat(
        (o_train.loc[:, 'column_name3':'column_name4'], o_test.loc[:, 'column_name3':'column_name4']))
    numeric_feats = all_labels.dtypes[all_labels.dtypes != "object"].index
    a = all_labels[numeric_feats].mean()
    b = all_labels[numeric_feats].std()
    return a, b


def amain():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    strat = time.perf_counter()
    # 读取原始数据
    o_train = pd.read_csv('your path/train.csv', encoding='utf-8')
    o_test = pd.read_csv('your path/test.csv', encoding='utf-8')

    # 自己的数据集，需要对原始数据进行处理
    # 对各维度的预处理(标准化)方式：数值型的转为[-1,1]之间 z-score 标准化，新数据=（原数据-均值）/标准差
    # 非数值型中的  无序型进行独热编码(one-hot encoding)，有序型 自己定义其数值 转换为数值型  本数据集默认全部为无序型
    # 空值：每一个特征的全局平均值来代替无效值

    # 将训练集与测试集的特征数据合并在一起 统一进行处理
    # loc：通过行标签索引数据 iloc：通过行号索引行数据 ix：通过行标签或行号索引数据（基于loc和iloc的混合）
    all_features = pd.concat((o_train.loc[:, 'column_name1':'column_name2'], o_test.loc[:, 'column_name1':'column_name2']))
    all_labels = pd.concat((o_train.loc[:, 'column_name3':'column_name4'], o_test.loc[:, 'column_name3':'column_name4']))

    # 对特征进行数据预处理
    # 对特征进行 z-score 标准化
    all_features = all_features.apply(lambda x: (x - x.mean()) / (x.std()))

    # 对标签值进行数据预处理
    # 取出所有的数值型特征名称
    numeric_feats = all_labels.dtypes[all_labels.dtypes != "object"].index
    object_feats = all_labels.dtypes[all_labels.dtypes == "object"].index

    # 将数值型特征进行 z-score 标准化
    all_labels[numeric_feats] = all_labels[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))

    # 对无序型进行one-hot encoding
    all_labels = pd.get_dummies(all_labels, prefix=object_feats, dummy_na=True)
    # 空值：每一个特征的全局平均值来代替无效值 NA就是指空值
    # all_labels = all_labels.fillna(all_labels.mean())

    num_train = o_train.shape[0]
    train_features = all_features[:num_train].values.astype(np.float32)
    test_features = all_features[num_train:].values.astype(np.float32)
    train_labels = all_labels[:num_train].values.astype(np.float32)
    test_labels = all_labels[num_train:].values.astype(np.float32)

    train_features = torch.from_numpy(train_features).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()
    # train_labels = torch.from_numpy(train_labels).unsqueeze(1).cuda()
    test_features = torch.from_numpy(test_features).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()
    # test_labels = torch.from_numpy(test_labels).unsqueeze(1).cuda()

    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)
    # 定义迭代器
    train_data = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
    test_data = DataLoader(dataset=test_set, batch_size=8, shuffle=False)

    input_num = train_features.shape[1]
    out_num = train_labels.shape[1]
    net = Net(input_num, out_num)
    net.to(device)  # 程序到指定机器上运行

    # 反向传播算法 SGD Adam等
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # 均方损失函数
    loss_function = torch.nn.MSELoss()

    # 记录用于绘图
    losses = []  # 记录每次迭代后训练的loss
    eval_losses = []  # 测试的
    save_path = 'your path/pre.pth'  # 权重保存路径
    epochs = 3000
    best_eval = 1.0  # 待会保存损失最低的模型
    a, b = label_mean_std()
    for epoch in range(epochs):
        train_loss = 0
        # train_acc = 0
        net.train()  # 网络设置为训练模式 暂时可加可不加
        for tdata, tlabel in train_data:
            # 前向传播
            output = net(tdata.to(device))
            # 记录单批次一次batch的loss
            loss = loss_function(output, tlabel.to(device))
            # loss = torch.sqrt(loss_function(output, tlabel.to(device)))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计单批次误差
            train_loss = train_loss + loss.item()

        losses.append(train_loss / len(train_data))
        # 测试集进行测试
        eval_loss = 0
        net.eval()  # 可加可不加
        with torch.no_grad():
            for edata, elabel in test_data:
                # 前向传播
                output = net(edata.to(device))
                # 记录单批次一次batch的loss，测试集就不需要反向传播更新网络了
                loss = loss_function(output, elabel.to(device))
                eval_loss = eval_loss + loss.item()

        eval_loss = eval_loss / len(test_data)

        eval_losses.append(eval_loss)

        print('epoch: {}, trainloss: {}, evalloss: {}'.format(epoch, losses[epoch], eval_losses[epoch]))

        if eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(net.state_dict(), save_path)

    # 测试最终模型的精准度 算一下测试集的平均误差
    end = time.perf_counter()
    print(end - strat)
    print(best_eval)

    iters = list(range(0, epochs))
    plt.plot(iters, eval_losses, linestyle='-', color='red', alpha=0.8, linewidth=1, label='loss')
    plt.title('300-500-relu')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    amain()
