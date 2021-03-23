# import config

# utils
import argparse
import csv
import datetime
import os
import random
import time

import torch
import numpy as np
from sklearn.metrics import roc_curve, mean_squared_error, mean_absolute_error, accuracy_score, auc
from torch import nn
import copy


def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)


def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len, seq_len), diagonal=-1).to(dtype=torch.bool)


def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0)


def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


#   MultiHead(Qin,Kin,Vin) = Concat(head1,··· ,headh)WO
class FFN(nn.Module):
    def __init__(self, features):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(features, features)
        self.layer2 = nn.Linear(features, features)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out = self.drop(self.relu(self.layer1(x)))
        out = self.layer2(out)
        return out


class MultiHeadWithFFN(nn.Module):
    def __init__(self, n_heads, n_dims, mask_type="ut", dropout=0.2):  # 这里实现的是Upper Trigger
        super(MultiHeadWithFFN, self).__init__()
        self.n_dims = n_dims
        self.mask_type = mask_type
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims, num_heads=n_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(n_dims)
        self.ffn = FFN(features=n_dims)
        self.layer_norm2 = nn.LayerNorm(n_dims)

    def forward(self, q_input, kv_input):
        q_input = q_input.permute(1, 0, 2)  # 为什么要转置呢，查过资料，这是输入的要求
        kv_input = kv_input.permute(1, 0, 2)  # 转置
        # LayerNorm(Q_in,K_in,V_in)
        query_norm = self.layer_norm1(q_input)  # 论文中并没有进行norm layer的处理，这里也许是作者加的
        kv_norm = self.layer_norm1(kv_input)
        if self.mask_type == "ut":
            mask = ut_mask(q_input.size(0))
        else:
            mask = lt_mask(q_input.size(0))
        if use_cuda:
            mask = mask.cuda()
        # multi head attention 输出有两个
        out_atten, weights_attent = self.multihead_attention(query=query_norm,
                                                             key=kv_norm,
                                                             value=kv_norm,
                                                             attn_mask=mask)
        out_atten += query_norm  # 这里是一个残差连接SkipConct
        out_atten = out_atten.permute(1, 0, 2)  # 这里需要进行变回来
        # LayerNorm(M)
        output_norm = self.layer_norm2(out_atten)  # 这里在进行一个layer norm
        # FFN(LayerNorm(M))
        output = self.ffn(output_norm)
        return output + output_norm  # SkipConct


# Encoder
class EncoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, total_ex, total_cat, seq_len):
        super(EncoderBlock, self).__init__()
        self.seq_len = seq_len
        self.exercise_embed = nn.Embedding(total_ex, n_dims)  # 输入的嵌入向量，这个我认为可以使用one hot 代替
        self.category_embed = nn.Embedding(total_cat, n_dims)  # 这个为什么使用类别嵌入，输入似乎不需要类别嵌入
        self.position_embed = nn.Embedding(seq_len, n_dims)  # 位置编码
        self.layer_norm = nn.LayerNorm(n_dims)  # Norm

        self.multihead = MultiHeadWithFFN(n_heads=n_heads, n_dims=n_dims)

    def forward(self, input_e, category, first_block=True):
        if first_block:  # 如果是第一个encoder sublayer
            _exe = self.exercise_embed(input_e)
            _cat = self.category_embed(category)
            position_encoded = pos_encode(self.seq_len).cuda()
            _pos = self.position_embed(position_encoded)
            out = _cat + _exe + _pos
        else:
            out = input_e
        output = self.multihead(q_input=out, kv_input=out)
        return output


# Decoder
class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, total_responses, seq_len):
        super(DecoderBlock, self).__init__()
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(total_responses, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.layer_norm = nn.LayerNorm(n_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                                         num_heads=n_heads,
                                                         dropout=0.2)
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=n_dims)

    def forward(self, input_r, encoder_output, first_block=True):
        if first_block:
            _response = self.response_embed(input_r)
            position_encoded = pos_encode(self.seq_len).cuda()
            _pos = self.position_embed(position_encoded)
            out = _response  + _pos
        else:
            out = input_r
        out = out.permute(1, 0, 2)
        # assert out_embed.size(0)==n_dims, "input dimention should be (seq_len,batch_size,dims)"
        out_norm = self.layer_norm(out)
        mask = ut_mask(out_norm.size(0))
        if use_cuda:
            mask = mask.cuda()
        out_atten, weights_attent = self.multihead_attention(query=out_norm,
                                                             key=out_norm,
                                                             value=out_norm,
                                                             attn_mask=mask)
        out_atten += out_norm
        out_atten = out_atten.permute(1, 0, 2)
        output = self.multihead(q_input=out_atten, kv_input=encoder_output)
        return output


# SAINT MODEL
class SAINT(nn.Module):
    def __init__(self, n_encoder, n_decoder, enc_heads, dec_heads, n_dims, total_ex, total_cat, total_responses,
                 seq_len):
        super(SAINT, self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.enocder = get_clones(EncoderBlock(enc_heads, n_dims, total_ex, total_cat, seq_len), n_encoder)
        self.decoder = get_clones(DecoderBlock(dec_heads, n_dims, total_responses, seq_len), n_decoder)
        self.fc = nn.Linear(n_dims, 1)

    def forward(self, in_exercise, in_category, in_response):
        first_block = True
        for n in range(self.n_encoder):
            if n >= 1:
                first_block = False

            enc = self.enocder[n](in_exercise, in_category, first_block=first_block)
            in_exercise = enc
            in_category = enc

        first_block = True
        for n in range(self.n_decoder):
            if n >= 1:
                first_block = False
            dec = self.decoder[n](in_response, encoder_output=in_exercise, first_block=first_block)
            in_exercise = dec
            in_response = dec

        return torch.sigmoid(self.fc(dec))


# 实验部分
# import
from torch.utils.data import Dataset, DataLoader

# variable
use_cuda = torch.cuda.is_available()
print("USE GPU:", use_cuda)
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


def collate(batch, q_num):
    # print("1",batch) # 列表：[[(题目，答案)，(题目，答案)，(题目，答案)……][(题目，答案)，……]……],32个包含一定数量(题目，答案)的列表
    lens = [len(row) for row in batch]
    # 最大题目数量
    max_len = 200 #max(lens)
    # 第三维的第三个数0或1表示数据是否有效，列表、元组前加星号是将其元素拆解为独立参数
    # benny：这里注意矩阵的加法操作，这里使用的是python语法，使用加法操作类似于append操作
    batch = tensor([[[*e, 1] for e in row] + [[0, 0, 0]] * (max_len - len(row)) for row in batch])
    Q, Y, S = batch.T  # Q:问题，Y:预测，S:padding,样本数据缺失或者说不够时填充[[0,0,0]]张量
    return Y.T, S.T, Q.T


def train(model, data, optimizer, batch_size):
    model.train(mode=True)
    # 创建一个测量目标和输出之间的二进制交叉熵的标准
    criterion = nn.BCELoss()
    if use_cuda:
        criterion = cuda(criterion)
    for Y, S, Q in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch, data.q_num),
            shuffle=True
    ):
        # in_exercise,in_cat,in_response
        #这里需要对Y进行一个平移，并且加入Start Token
        R = Y[:, :-1]
        r_batch_size, r_lenght = R.shape
        start_token = cuda(torch.full([r_batch_size, 1], 2, dtype=torch.int64))  # 使用 2 作为 start token
        R = torch.cat([start_token, R], dim=1)
        if use_cuda:
            Q = cuda(Q)
            Y = cuda(Y)
            R = cuda(R)
        P = model(Q, Q, R)

        index = S == 1

        batch_size,length,feature = P.shape
        P = torch.reshape(P,(batch_size,length))
        # 损失函数计算
        loss = criterion(P[index], Y[index].float())
        # 把模型中参数的梯度设为0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 一步梯度下降，执行单个优化步骤（参数更新）
        optimizer.step()


def evaluate(model, data, batch_size):
    model.eval()
    criterion = nn.BCELoss()
    y_pred, y_true = [], []
    loss = 0.0
    for Y, S, Q in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch, data.q_num)
    ):
        #这里需要对Y进行一个平移，并且加入Start Token
        R = Y[:, :-1]
        r_batch_size, r_lenght = R.shape
        start_token = cuda(torch.full([r_batch_size, 1], 2, dtype=torch.int64))  # 使用 2 作为 start token
        R = torch.cat([start_token, R], dim=1)
        if use_cuda:
            Q = cuda(Q)
            Y = cuda(Y)
            R = cuda(R)
        P = model(Q, Q, R)
        index = S == 1
        batch_size,length,feature = P.shape
        P = torch.reshape(P,(batch_size,length))
        P, Y = P[index], Y[index].float()
        # 截断反向传播的梯度流，不需要计算梯度
        y_pred += detach(P)
        y_true += detach(Y)
        # P = torch.sigmoid(P)
        # 计算32个样本损失和
        loss += detach(criterion(P, Y))

    # fpr:假阳性率;tpr:真阳性率;thres:减少了用于计算fpr和tpr的决策函数的阈值.
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    # auc, loss, mse, acc
    return auc(fpr, tpr), loss, mse_value, mae_value, acc_value


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


def experiment(
        dataset,
        hidden_num,
        learning_rate,
        length,
        epochs,
        batch_size,
        seed,
        q_num,
        cv_num,
        n_encoder,
        n_decoder,
        enc_heads,
        dec_heads,
        n_dims,
        total_ex,
        total_cat,
        total_responses,
        seq_len

):
    # 设置随机数生成器的种子
    set_seed(seed)
    # Data实例化:test_data
    test_data = Data(open('./my_data/%s/test.csv' % dataset, 'r'), length, q_num, is_test=True)
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
        # 1238:train_valid.csv 1238是在该数据集中有1238个学生，在做交叉验证的的时候选出20%的学生作为验证集
        origin_list = [i for i in range(3360)]
        # valid set index
        # 使随机数生成器的种子不同，从而使每次迭代分割出不同的验证集
        random.seed(cv + 1000)
        # random.sample(range(100), 10)：从100个数中不重复随机抽样10个数，这里是五倍交叉验证
        index_split = random.sample(origin_list, int(0.2 * len(origin_list)))
        random.seed(0)

        train_data = Data(open('./my_data/%s/train_valid.csv' % dataset, 'r'), length, q_num, is_test=False,
                          index_split=index_split, is_train=True)
        valid_data = Data(open('./my_data/%s/train_valid.csv' % dataset, 'r'), length, q_num, is_test=False,
                          index_split=index_split, is_train=False)
        max_auc = 0.0
        # DKT模型实例化：model
        model = cuda(SAINT(n_encoder,
                           n_decoder,
                           enc_heads,
                           dec_heads,
                           n_dims,
                           total_ex,
                           total_cat,
                           total_responses,
                           seq_len))

        # torch.optim.SGD：类，实现随机梯度下降（可选带动量）的优化器,params：用于优化的参数迭代或定义参数组的dicts，momentum:动量因子
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
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


# 数据集：assist2009, synthetic, assist2015, STATICS，assist2012，eanalyst
# 创建解析步骤
parser = argparse.ArgumentParser(description='Script to test DKT.')
# 添加参数步骤
parser.add_argument('--dataset', type=str, default='ccnu', help='')
parser.add_argument('--hidden_num', type=int, default=128, help='')
parser.add_argument('--learning_rate', type=float, default=0.09, help='')
parser.add_argument('--length', type=int, default=200, help='')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--q_num', type=int, default=2750, help='')
parser.add_argument('--cv_num', type=int, default=5, help='')
# SAINT ADDED
parser.add_argument('--n_encoder', type=int, default=6, help='')
parser.add_argument('--n_decoder', type=int, default=6, help='')
parser.add_argument('--enc_heads', type=int, default=8, help='')
parser.add_argument('--dec_heads', type=int, default=8, help='')
parser.add_argument('--n_dims', type=int, default=128, help='')
parser.add_argument('--total_ex', type=int, default=2750, help='')
parser.add_argument('--total_cat', type=int, default=2750, help='')
parser.add_argument('--total_responses', type=int, default=3, help='')
parser.add_argument('--seq_len', type=int, default=200, help='')

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
    n_encoder=params.n_encoder,
    n_decoder=params.n_decoder,
    enc_heads=params.enc_heads,
    dec_heads=params.dec_heads,
    n_dims=params.n_dims,
    total_ex=params.total_ex,
    total_cat=params.total_cat,
    total_responses=params.total_responses,
    seq_len=params.seq_len
)
