from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import csv #读 CSV
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader


def get_feature_importance(feature_data, label_data, k =4,column = None):
    """
    feature_data, label_data 要求字符串形式
    k为选择的特征数量
    如果需要打印column，需要传入行名
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    这个函数的目的是， 找到所有的特征种， 比较有用的k个特征， 并打印这些列的名字。
    """
    model = SelectKBest(chi2, k=k)      #定义一个选择k个最佳特征的函数
    feature_data = np.array(feature_data, dtype=np.float64)
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]        #[::-1]表示反转一个列表或者矩阵。
    # argsort这个函数， 可以矩阵排序后的下标。 比如 indices[0]表示的是，scores中最小值的下标。

    if column:                            # 如果需要打印选中的列名字
        k_best_features = [column[i] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。

class covidDataset(Dataset):
    def __init__(self, file_path, mode, dim=4, all_feature=False):
        with open(file_path, "r") as f:
            csv_data = list(csv.reader(f))
            data = np.array(csv_data[1:])              # 1: 第一行后面的
            if mode == "train":                      # 训练数据逢5选4， 记录他们的所在行
                indices = [i for i in range(len(data)) if i % 5 !=0]          #1，2，3，4， 6，7，8，9
            elif mode == "val":                           # 验证数据逢5选1， 记录他们的所在列
                indices = [i for i in range(len(data)) if i % 5 ==0]

            if all_feature:
                col_idx = [i for i in range(0,93)]       # 若全选，则选中所有列。
            else:
                _, col_idx = get_feature_importance(data[:,1:-1], data[:,-1], k=dim,column =csv_data[0][1:-1]) # 选重要的dim列。


            if mode == "test":
                x = data[:, 1:].astype(float)          #测试集没标签，取第二列开始的数据，并转为float
                x = torch.tensor(x[:, col_idx])              #  col_idx表示了选取的列，转为张量
            else:
                x = data[indices, 1:-1].astype(float)
                x = torch.tensor(x[:, col_idx])
                y = data[indices, -1].astype(float)      #训练接和验证集有标签，取最后一列的数据，并转为float
                self.y = torch.tensor(y)              #转为张量
            self.x = (x-x.mean(dim=0,keepdim=True))/x.std(dim=0,keepdim=True)        # 对数据进行列归一化 0正太分布
            self.mode = mode              # 表示当前数据集的模式
    def __getitem__(self, item):
        if self.mode == "test":
            return self.x[item].float()         # 测试集没标签。   注意data要转为模型需要的float32型
        else:                            # 否则要返回带标签数据
            return self.x[item].float(), self.y[item].float()
    def __len__(self):
        return len(self.x)             # 返回数据长度。


class myNet(nn.Module):
    def __init__(self, inDim):
        super(myNet,self).__init__()
        self.fc1 = nn.Linear(inDim, 128)              # 全连接
        self.relu = nn.ReLU()                        # 激活函数 ,添加非线性
        # self.fc3 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128,1)                     # 全连接             设计模型架构。 他没有数据

    def forward(self, x):                     #forward， 即模型前向过程
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc3(x)
        x = self.fc2(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x




def train_val(model, trainloader, valloader,optimizer, loss, epoch, device, save_):

    # trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    # valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    model = model.to(device)                # 模型和数据 ，要在一个设备上。  cpu - gpu
    plt_train_loss = []
    plt_val_loss = []
    val_rel = []
    min_val_loss = 100000                 # 记录训练验证loss 以及验证loss和结果

    for i in range(epoch):                 # 训练epoch 轮
        start_time = time.time()             # 记录开始时间
        model.train()                         # 模型设置为训练状态      结构
        train_loss = 0.0
        val_loss = 0.0
        for data in trainloader:                     # 从训练集取一个batch的数据
            optimizer.zero_grad()                   # 梯度清0
            x, target = data[0].to(device), data[1].to(device)       # 将数据放到设备上
            pred = model(x)                          # 用模型预测数据
            bat_loss = loss(pred, target)       # 计算loss
            bat_loss.backward()                        # 梯度回传， 反向传播。
            optimizer.step()                            #用优化器更新模型。  轮到SGD出手了
            train_loss += bat_loss.detach().cpu().item()             #记录loss和

        plt_train_loss. append(train_loss/trainloader.__len__())   #记录loss到列表。注意是平均的loss ，因此要除以数据集长度。

        model.eval()                 # 模型设置为验证状态
        with torch.no_grad():                    # 模型不再计算梯度
            for data in valloader:                      # 从验证集取一个batch的数据
                val_x , val_target = data[0].to(device), data[1].to(device)          # 将数据放到设备上
                val_pred = model(val_x)                 # 用模型预测数据
                val_bat_loss = loss(val_pred, val_target)          # 计算loss
                val_loss += val_bat_loss.detach().cpu().item()                  # 计算loss
                val_rel.append(val_pred)                 #记录预测结果
        if val_loss < min_val_loss:
            torch.save(model, save_)               #如果loss比之前的最小值小， 说明模型更优， 保存这个模型

        plt_val_loss.append(val_loss/valloader.dataset.__len__())  #记录loss到列表。注意是平均的loss ，因此要除以数据集长度。
        #
        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f' % \
              (i, epoch, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1])
              )              #打印训练结果。 注意python语法， %2.2f 表示小数位为2的浮点数， 后面可以对应。


        # print('[%03d/%03d] %2.2f sec(s) TrainLoss : %3.6f | valLoss: %.6f' % \
        #       (i, epoch, time.time()-start_time, 2210.2255411, plt_val_loss[-1])
        #       )              #打印训练结果。 注意python语法， %2.2f 表示小数位为2的浮点数， 后面可以对应。
    plt.plot(plt_train_loss)              # 画图， 向图中放入训练loss数据
    plt.plot(plt_val_loss)                # 画图， 向图中放入训练loss数据
    plt.title('loss')                      # 画图， 标题
    plt.legend(['train', 'val'])             # 画图， 图例
    plt.show()                                 # 画图， 展示





def evaluate(model_path, testset, rel_path ,device):
    model = torch.load(model_path).to(device)                     # 模型放到设备上。  加载模型
    testloader = DataLoader(testset, batch_size=1, shuffle=False)         # 将验证数据放入loader 验证时， 一般batch为1
    val_rel = []
    model.eval()               # 模型设置为验证状态
    with torch.no_grad():               # 模型不再计算梯度
        for data in testloader:                 # 从测试集取一个batch的数据
            x = data.to(device)                # 将数据放到设备上
            pred = model(x)                        # 用模型预测数据
            val_rel.append(pred.item())                #记录预测结果
    print(val_rel)                                     #打印预测结果
    with open(rel_path, 'w') as f:                        #打开保存的文件
        csv_writer = csv.writer(f)                           #初始化一个写文件器 writer
        csv_writer.writerow(['id','tested_positive'])         #在第一行写上 “id” 和 “tested_positive”
        for i in range(len(testset)):                           # 把测试结果的每一行放入输出的excel表中。
            csv_writer.writerow([str(i),str(val_rel[i])])
    print("rel已经保存到"+ rel_path)





all_col = False            #是否使用所有的列
device = 'cuda' if torch.cuda.is_available() else 'cpu'       #选择使用cpu还是gpu计算。
print(device)
train_path = 'covid.train.csv'                     # 训练数据路径
test_path = 'covid.test.csv'              # 测试数据路径
file = pd.read_csv(train_path)
file.head()                    # 用pandas 看看数据长啥样

if all_col == True:
    feature_dim = 93
else:
    feature_dim = 6              #是否使用所有的列

trainset = covidDataset(train_path,'train', feature_dim, all_feature=all_col)
valset = covidDataset(train_path,'val', feature_dim, all_feature=all_col)
testset = covidDataset(test_path,'test', feature_dim, all_feature=all_col)   #读取训练， 验证，测试数据

         # 返回损失。
#
# def mseLoss(pred, target, model):
#     loss = nn.MSELoss(reduction='mean')
#     ''' Calculate loss '''
#     regularization_loss = 0                    # 正则项
#     for param in model.parameters():
#         # TODO: you may implement L1/L2 regularization here
#         # 使用L2正则项
#         # regularization_loss += torch.sum(abs(param))
#         regularization_loss += torch.sum(param ** 2)                  # 计算所有参数平方
#     return loss(pred, target) + 0.00075 * regularization_loss             # 返回损失。
#
# loss =  mseLoss           # 定义mseloss 即 平方差损失，


loss =  nn.MSELoss()          # 定义mseloss 即 平方差损失，

config = {
    'n_epochs': 50,                # maximum number of epochs
    'batch_size': 32,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
            # hyper-parameters for the optimizer (depends on which optimizer you are using)
    'lr': 0.0001,                 # learning rate of SGD
    'momentum': 0.9,             # momentum for SGD
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'model_save/model.pth',  # your model will be saved here
}

model = myNet(feature_dim).to(device)                      # 实例化模型

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)             # 定义优化器  动量
trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=True)  # 将数据装入loader 方便取一个batch的数据

train_val(model, trainloader, valloader, optimizer, loss, config['n_epochs'], device,save_=config['save_path'])  # 训练


evaluate(config['save_path'], testset, 'pred.csv', device)           # 验证



















