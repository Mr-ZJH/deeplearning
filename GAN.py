import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

matplotlib_is_available = True
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("Will skip plotting; matplotlib is not available.")
    matplotlib_is_available = False


# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(11, 50),  # 用线性变换将输入映射到50维
            nn.ReLU(True),  # relu激活
            nn.Linear(50, 100),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(100, 11),  # 线性变换
            nn.Tanh())  # Tanh激活使得生成数据分布在【-1,1】之间

    def forward(self, x):
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(11, 50),  # 输入特征数为11，输出为50
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(50, 100),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid())  # 也是一个激活函数，二分类问题中，
        # sigmoid可以班实数映射到【0,1】，作为概率值，
        # 多分类用softmax函数

    def forward(self, x):
        x = self.dis(x)
        return x


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 3
    d_learning_rate = 0.002
    g_learning_rate = 0.002

    num_epochs = 1500

    trust_data = pd.read_csv('your data path', encoding='utf-8')
    # 归一化数据
    trust_data = trust_data.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    trust_data = trust_data.values.astype(np.float32)
    trust_data = torch.from_numpy(trust_data).cuda()
    trust_data = DataLoader(dataset=trust_data, batch_size=batch_size, shuffle=True)

    G = Generator().cuda()
    D = Discriminator().cuda()
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)

    d_losses = []
    g_losses = []

    for epoch in range(num_epochs):
        for i, data in enumerate(trust_data):
            num = len(data)
            dimension = len(data[0])
            # 1. Train D on real+fake

            real_label = Variable(torch.ones(num)).cuda()  # 定义真实的数据的label为1
            fake_label = Variable(torch.zeros(num)).cuda()  # 定义假的数据的label为0
            #  1A: Train D on real
            d_real_data = Variable(data).cuda()
            d_real_decision = D(d_real_data)
            d_real_decision = d_real_decision.squeeze(axis=1)  # (8,1) -> (8,)
            d_real_error = criterion(d_real_decision, real_label)  # ones = true
            real_scores = d_real_decision  # 得到真实数据的判别值，输出的值越接近1越好

            #  1B: Train D on fake
            d_gen_input = Variable(torch.randn(num, dimension)).to(device)
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            d_fake_decision = d_fake_decision.squeeze(axis=1)  # (8,1) -> (8,)
            d_fake_error = criterion(d_fake_decision, fake_label)  # zeros = fake
            fake_scores = d_fake_decision  # 得到假数据的判别值，对于判别器来说，假数据的损失越接近0越好
            # 损失函数和优化
            d_loss = d_real_error + d_fake_error  # 损失包括判真损失和判假损失
            d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            d_loss.backward()  # 将误差反向传播
            d_optimizer.step()  # 更新参数

            # 2. Train G on D's response (but DO NOT train D on these labels)
            gen_input = Variable(torch.randn(num, dimension)).cuda()
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            dg_fake_decision = dg_fake_decision.squeeze(axis=1)  # (8,1) -> (8,)
            g_loss = criterion(dg_fake_decision, real_label)  # Train G to pretend it's genuine
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()
            g_optimizer.step()  # Only optimizes G's parameters

            # 打印中间的损失
            if i % batch_size == 0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                      'D real: {:.6f},D fake: {:.6f}'.format(
                    epoch, num_epochs, d_loss.data.item(), g_loss.data.item(),
                    real_scores.data.mean(), fake_scores.data.mean()))  # 打印的是真实数据的损失均值

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    torch.save(G.state_dict(), './generator.pth')  # 保存generator权重
    torch.save(D.state_dict(), './discriminator.pth')  # 保存discriminator权重

    if matplotlib_is_available:
        print("Plotting the generated distribution...")
        # values = g_fake_data.detach().cpu().numpy()
        # print(" Values: %s" % (str(values)))
        # plt.hist(values, bins=50)
        # plt.xlabel('Value')
        # plt.ylabel('Count')
        # plt.title('Histogram of Generated Distribution')
        # plt.grid(True)
        # plt.show()

        fig, ax = plt.subplots(2, 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        ax[0].plot(range(len(d_losses)), d_losses)
        ax[0].set_ylabel('d_loss')
        ax[0].set_title('Discriminator')
        ax[1].plot(range(len(g_losses)), g_losses)
        ax[1].set_ylabel('g_loss')
        ax[1].set_title('Generator')
        plt.show()


if __name__ == '__main__':
    train()
