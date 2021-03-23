#!/usr/bin/env python
# coding: utf-8


# 加了五倍交叉验证的DKT
import os
import datetime
import random
import time
import csv
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score
# sklearn.metrics:包含了许多模型评估指标，例如决定系数R2、准确度等;
# roc_curve:roc曲线；mean_squared_error：均方差；mean_absolute_error：平均绝对误差；accuracy_score:准确率
import argparse
# 用于从 sys.argv 中解析命令项选项与参数的模块
import numpy as np

# ### 定义相关函数

# In[10]:


# 是否适用gpu，cuda()用于将变量传输到GPU上，gpu版本是torch.cuda.FloatTensor,cpu版本是torch.FloatTensor
use_cuda = torch.cuda.is_available()
print("USE GPU:",use_cuda)
device = 0
if use_cuda:
    torch.cuda.set_device(device)
# 定义cuda()函数：参数为o，如果use_cuda为真返回o.cuda(),为假返回o
cuda = lambda o: o.cuda() if use_cuda else o
# torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。
tensor = lambda o: cuda(torch.tensor(o))
# 生成对角线全1，其余部分全0的二维数组,函数原型：torch.eye(n, m=None, out=None)，m (int) ：列数.如果为None,则默认为n。
eye = lambda d: cuda(torch.eye(d))
# 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor。
zeros = lambda *args: cuda(torch.zeros(*args))

# 截断反向传播的梯度流,返回一个新的Variable即tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
# 不同之处只是它的requires_grad是false，也就是说这个Variable永远不需要计算其梯度，不具有grad。
detach = lambda o: o.cpu().detach().numpy().tolist()


def set_seed(seed=0):
    # seed()方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数。random:随机数生成器，seed:种子
    random.seed(seed)
    # 为CPU设置种子用于生成随机数
    torch.manual_seed(seed)
    # 为当前GPU设置随机种子,如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed(seed)
    '''
    置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
    应该可以保证每次运行网络的时候相同输入的输出是固定的。（说人话就是让每次跑出来的效果是一致的）
    '''
    torch.backends.cudnn.deterministic = True
    '''
     置为True的话会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    '''
    torch.backends.cudnn.benchmark = False


# ### 加载数据集

# In[11]:


# 读取数据集
# 数据集情况：题目序列的长度 题目序列 答对的情况
class Data:

    def __init__(self, file, length, q_num, is_test=False, index_split=None, is_train=False):
        '''
        len: 4
        q: 53,54,53,54
        y: 0,0,0,0
        t1: 0,1,2,0
        t2: 0,1,3,5
        t3: 3,1,2,1
        '''
        # 读取csv文件，delimiter说明分割字段的字符串为逗号
        rows = csv.reader(file, delimiter=',')
        # rows为:[[题目个数], [题目序列], [答对情况]……]
        rows = [[int(e) for e in row if e != ''] for row in rows]

        q_rows, r_rows = [], []

        student_num = 0
        # zip()将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象,注意：不是列表
        if is_test:
            # 双冒号：实质为lis[start:end:step]，end没写就是到列表的最后，意思就是从索引为start到end，步长为step进行切片，每个step取一次
            # q_row, r_row：题目序号列表，答对情况列表
            for q_row, r_row in zip(rows[1::3], rows[2::3]):
                num = len(q_row)
                n = num // length
                for i in range(n + 1):
                    q_rows.append(q_row[i * length: (i + 1) * length])
                    r_rows.append(r_row[i * length: (i + 1) * length])
        else:
            if is_train:
                for q_row, r_row in zip(rows[1::3], rows[2::3]):

                    if student_num not in index_split:

                        num = len(q_row)

                        n = num // length

                        for i in range(n + 1):
                            q_rows.append(q_row[i * length: (i + 1) * length])

                            r_rows.append(r_row[i * length: (i + 1) * length])
                    student_num += 1
            # 验证集
            else:
                for q_row, r_row in zip(rows[1::3], rows[2::3]):

                    if student_num in index_split:

                        num = len(q_row)

                        n = num // length

                        for i in range(n + 1):
                            q_rows.append(q_row[i * length: (i + 1) * length])

                            r_rows.append(r_row[i * length: (i + 1) * length])
                    student_num += 1

        q_rows = [row for row in q_rows if len(row) > 2]

        r_rows = [row for row in r_rows if len(row) > 2]

        # q_min = min([min(row) for row in q_rows])

        # q_rows = [[q - q_min for q in row] for row in q_rows]

        self.r_rows = r_rows

        # self.q_num = max([max(row) for row in q_rows]) + 1
        self.q_num = q_num
        self.q_rows = q_rows

    # 获取[[题号,答对]，[题号,答对]，……]列表
    def __getitem__(self, index):
        return list(
            zip(self.q_rows[index], self.r_rows[index]))

    # 批次大小
    def __len__(self):
        return len(self.q_rows)


# ### 处理样本数据（将input处理成合适的维度或shape）

# In[19]:


def collate(batch, q_num):
    # print("1",batch) # 列表：[[(题目，答案)，(题目，答案)，(题目，答案)……][(题目，答案)，……]……],32个包含一定数量(题目，答案)的列表
    lens = [len(row) for row in batch]
    # 最大题目数量
    max_len = max(lens)
    batch = tensor([[[*e, 1] for e in row] + [[0, 0, 0]] * (max_len - len(row)) for row in batch])
    Q, Y, S = batch.T  # Q:问题，Y:预测，S:padding,样本数据缺失或者说不够时填充[[0,0,0]]张量
    Q, Y, S = Q.T, Y.T, S.T  # torch.size([32,200])
    X = Q + q_num * Y  # 由于类别不只是0 和 1 了，所以加上Y才正确
    return X, Y, S, Q


# ### DKT模型

# In[13]:


class DKT(nn.Module):
    def __init__(self, q_num, h_num):
        # 调用父类，解决多重继承问题
        super(DKT, self).__init__()
        drop_prob1, drop_prob2 = 0.2, 0.4
        # 隐藏状态的特征数
        self.h_num = h_num
        # q_num：处理过后训练集的题目个数，
        print("q_num=", q_num)
        # ???
        self.x_onehot = eye(3 * q_num)  # 现在是一个二分类问题所以需要乘以2
        self.q_onehot = eye(q_num)
        #self.rnn = nn.RNN(q_num * 3, h_num, num_layers=2, batch_first=True)
        self.rnn = nn.LSTM(q_num * 3,h_num,num_layers=2,batch_first=True) # 可以更换为LSTM来减少梯度消失
        # 用于分类的RNN
#         self.class_rnn = nn.RNN(input_size=1, hidden_size=h_num, num_layers=2,batch_first=True)
        self.class_rnn_fc = nn.Sequential(
            # 激活函数
            #nn.Tanh(),
            nn.Linear(1, 3),
            #nn.ReLU(),
            #nn.Linear(3,3),
            #nn.ReLU(),
            #nn.Softmax(dim=2),
        )
        self.FCs = nn.Sequential(
            # 激活函数
            nn.Tanh(),
            nn.Linear(h_num, q_num),
            nn.Sigmoid()
        )

    def forward(self, X,Q):
        size, length = X.shape  # X.shape:(32,200),X为二维张量，来自于collate函数
        # onehot编码
        X = self.x_onehot[X]
        X, hidden = self.rnn(X, (zeros(2, size, self.h_num),zeros(2, size, self.h_num)))
        #X, hidden = self.rnn(X, zeros(2, size, self.h_num))
        P = self.FCs(X)
        Q = self.q_onehot[Q]
        Q, P = Q[:, 1:], P[:, :-1]
        class_rnn_input = (Q * P).sum(2)
        class_rnn_input = torch.reshape(class_rnn_input,(size,length - 1,1)) # 32,199,1
        #out,class_hidden = self.class_rnn(class_rnn_input,zeros(2,size,self.h_num))
        out = self.class_rnn_fc(class_rnn_input)
       
        
        return out


# ### 训练模型+验证+测试

# In[1]:


def train(model, data, optimizer, batch_size):
    '''
    使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval。
    在model(test)之前，需要加上model.eval()，框架会自动把BN(BatchNormalization)和DropOut固定住，不会取平均，而是用训练好的值。
    否则的话，有输入数据（test的batch_size过小，），即使不训练，它也会改变权值，这是model中含有batch normalization层所带来的的性质。
    model.train()：启用 BatchNormalization 和 Dropout
    model.eval()：不启用 BatchNormalization 和 Dropout
    '''
    model.train(mode=True)
    # 创建一个测量目标和输出之间的二进制交叉熵的标准
    criterion = nn.CrossEntropyLoss()  # 多分类问题中
    #criterion = nn.BCELoss()
    for X, Y, S, Q in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch, data.q_num),
            shuffle=True
    ):
        Y_HAT = model(X,Q)
        Y, S = Y[:, 1:], S[:, 1:]
        index = S == 1
        dkt_y_hat = Y_HAT[index]
        y = Y[index]

        # 损失函数计算
        loss = criterion(dkt_y_hat,y)

        # 把模型中参数的梯度设为0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 一步梯度下降，执行单个优化步骤（参数更新）
        optimizer.step()




def evaluate(model, data, batch_size):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    y_pred, y_true = [], []
    loss = 0.0
    for X, Y, S, Q in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch, data.q_num)
    ):

        Y_HAT = model(X, Q)
        Y, S = Y[:, 1:], S[:, 1:]
        index = S == 1
        dkt_y_hat = Y_HAT[index]
        y = Y[index]

        y_pred += detach(dkt_y_hat)
        y_true += detach(y)
        loss += detach(criterion(dkt_y_hat, y))

    # fpr:假阳性率;tpr:真阳性率;thres:减少了用于计算fpr和tpr的决策函数的阈值.
    y_pred = [np.argmax(i) for i in y_pred]
    #print("y pred:", y_pred)
    #print("y true:", y_true)
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    # bi_y_pred = [torch.argmax(i) for i in y_pred]
    acc_value = accuracy_score(y_true, y_pred)
    # auc, loss, mse, acc
    return auc(fpr, tpr), loss, mse_value, mae_value, acc_value


# 参数　dataset：eanalyst_math/eanalyst_math；hidden_num：128；learning_rate：0.09；length：200,epochs：200,batch_size：32,seed：0,# embed_dim,
# 　q_num：2750,cv_num：5

def experiment(
        dataset,
        hidden_num,
        # concept_num,
        learning_rate,
        length,
        epochs,
        batch_size,
        seed,
        # embed_dim,
        q_num,
        cv_num,
data_cross_all
):
    # 设置随机数生成器的种子
    set_seed(seed)
    # Data实例化:test_data
    test_data = Data(open('./my_data/%s/builder_test.csv' % dataset, 'r'), length, q_num, is_test=True)
    path = './result_dkt_cross/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))
    os.makedirs(path)
    info_file = open('%s/info.txt' % path, 'w+')

    params_list = (
        'dataset = %s\n' % dataset,
        'hidden_size = %d\n' % hidden_num,
        # 'concept_num = %d\n' % concept_num,
        'learning_rate = %f\n' % learning_rate,
        'length = %d\n' % length,
        'batch_size = %d\n' % batch_size,
        'seed = %d\n' % seed,
        'q_num = %d\n' % q_num
    )
    info_file.write('file_name = allxt-onehot no norm + weight decay 5e-4')
    info_file.write('%s%s%s%s%s%s%s' % params_list)

    total_auc = 0.0
    model_list = []

    for cv in range(cv_num):
        # ccnu 数据集里有522个学生
        origin_list = [i for i in range(data_cross_all)]
        # valid set index
        # 使随机数生成器的种子不同，从而使每次迭代分割出不同的验证集
        random.seed(cv + 1000)
        # random.sample(range(100), 10)：从100个数中不重复随机抽样10个数，这里是五倍交叉验证
        index_split = random.sample(origin_list, int(0.2 * len(origin_list)))
        random.seed(0)

        train_data = Data(open('./my_data/%s/builder_train.csv' % dataset, 'r'), length, q_num, is_test=False,
                          index_split=index_split, is_train=True)
        valid_data = Data(open('./my_data/%s/builder_train.csv' % dataset, 'r'), length, q_num, is_test=False,
                          index_split=index_split, is_train=False)
        max_auc = 0.0
        # DKT模型实例化：model
        model = cuda(DKT(train_data.q_num, hidden_num))

        # torch.optim.SGD：类，实现随机梯度下降（可选带动量）的优化器,params：用于优化的参数迭代或定义参数组的dicts，momentum:动量因子
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)

        for epoch in range(1, epochs + 1):
            time_start = time.time()
            train(model, train_data, optimizer, batch_size)
            train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model, train_data, batch_size)
            valid_auc, valid_loss, valid_mse, valid_mae, valid_acc = evaluate(model, valid_data, batch_size)
            time_end = time.time()

            if max_auc < valid_auc:
                max_auc = valid_auc
                torch.save(model.state_dict(), '%s/model_%s' % ('%s' % path, '%d' % cv))
                current_max_model = model

            print_list = (
                'cv:%-3d' % cv,
                'epoch:%-3d' % epoch,
                'max_auc:%-8.4f' % max_auc,
                'valid_auc:%-8.4f' % valid_auc,
                'valid_loss:%-8.4f' % valid_loss,
                'valid_mse:%-8.4f' % valid_mse,
                'valid_mae:%-8.4f' % valid_mae,
                'valid_acc:%-8.4f' % valid_acc,
                'train_auc:%-8.4f' % train_auc,
                'train_loss:%-8.4f' % train_loss,
                'train_mse:%-8.4f' % train_mse,
                'train_mae:%-8.4f' % train_mae,
                'train_acc:%-8.4f' % train_acc,
                'time:%-6.2fs' % (time_end - time_start)
            )

            print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % print_list)
            info_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list)
        model_list.append(current_max_model)

    # 模型测试
    train_list = []
    auc_list = []
    mse_list = []
    mae_list = []
    acc_list = []
    loss_list = []
    # enumerate() 函数:用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出索引和序列元素，一般用在 for 循环当中。
    for cv, model_item in enumerate(model_list):
        train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model_item, train_data, batch_size)
        test_auc, test_loss, test_mse, test_mae, test_acc = evaluate(model_item, test_data, batch_size)

        train_list.append(train_auc)
        auc_list.append(test_auc)
        mse_list.append(test_mse)
        mae_list.append(test_mae)
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        print_list_test = (
            'cv:%-3d' % cv,
            'train_auc:%-8.4f' % train_auc,
            'test_auc:%-8.4f' % test_auc,
            'test_mse:%-8.4f' % test_mse,
            'test_mae:%-8.4f' % test_mae,
            'test_acc:%-8.4f' % test_acc,
            'test_loss:%-8.4f' % test_loss
        )

        print('%s %s %s %s %s %s %s\n' % print_list_test)
        info_file.write('%s %s %s %s %s %s %s\n' % print_list_test)

    average_train_auc = sum(train_list) / len(train_list)
    average_test_auc = sum(auc_list) / len(auc_list)
    average_test_mse = sum(mse_list) / len(mse_list)
    average_test_mae = sum(mae_list) / len(mae_list)
    average_test_acc = sum(acc_list) / len(acc_list)
    average_test_loss = sum(loss_list) / len(loss_list)
    print_result = (
        'average_train_auc:%-8.4f' % average_train_auc,
        'average_test_auc:%-8.4f' % average_test_auc,
        'average_test_mse:%-8.4f' % average_test_mse,
        'average_test_mae:%-8.4f' % average_test_mae,
        'average_test_acc:%-8.4f' % average_test_acc,
        'average_test_loss:%-8.4f' % average_test_loss
    )
    print('%s %s %s %s %s %s\n' % print_result)
    info_file.write('%s %s %s %s %s %s\n' % print_result)


# ### 运行程序-使用命令行参数传入相关参数

# In[ ]:


# 数据集：assist2009, synthetic, assist2015, STATICS，assist2012，eanalyst
# 创建解析步骤
parser = argparse.ArgumentParser(description='Script to test DKT.')
# 添加参数步骤
parser.add_argument('--dataset', type=str, default='assistments', help='')
parser.add_argument('--hidden_num', type=int, default=128, help='')
parser.add_argument('--learning_rate', type=float, default=0.09, help='')
parser.add_argument('--length', type=int, default=200, help='')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--q_num', type=int, default=130, help='')
parser.add_argument('--cv_num', type=int, default=5, help='')
parser.add_argument('--data_cross_all', type=int, default=3360, help='')
# 参数中加入args=[]
params = parser.parse_args(args=[])

experiment(
    dataset=params.dataset,
    hidden_num=params.hidden_num,
    learning_rate=params.learning_rate,
    length=params.length,
    epochs=params.epochs,
    batch_size=params.batch_size,
    seed=params.seed,
    q_num=params.q_num,
    cv_num=params.cv_num,
    data_cross_all=params.data_cross_all
)
