import matplotlib.pyplot as plt
import torch
import numpy as np
import csv
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import time
from sklearn.feature_selection import SelectKBest, chi2


# 这个方法用于筛选有用的特征列
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
    feature_data = np.array(feature_data, dtype=np.float64)  # x转换为矩阵
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征，这个函数起到主要作用
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]        #[::-1]表示反转一个列表或者矩阵。
    # argsort这个函数， 可以矩阵排序后的下标。 比如 indices[0]表示的是，scores中最小值的下标。
    # 排序后输出矩阵的下标

    if column:                            # 如果需要打印选中的列名字
        k_best_features = [column[i] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。



class CovidDataset(Dataset):
    def __init__(self, file_path, mode,all_feature=True,feature_dim=6):
        with open(file_path, "r") as f:
            # 读取原始数据
            ori_data = list(csv.reader(f))
            # 原始数据去掉第一行第一列并转化为浮点型数据
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)
            self.mode = mode
            column=ori_data[0]
            # 以下是对提取的数据进行改进，使用get_feature_importance提取有关的特征
            feature = np.array(ori_data[1:])[:,1:-1]
            label_data = np.array(ori_data[1:])[:,-1]
            if all_feature:
                col = [i for i in range(0,93)]
            else:
                _,col = get_feature_importance(feature, label_data,feature_dim,column=column)
            col = col.tolist()

            if mode == "train":
                incices = [i for i in range(len(csv_data)) if i % 5 != 0]
                self.y = torch.tensor(csv_data[incices, -1])
                data = torch.tensor(csv_data[incices, : -1])
            elif mode == "val":
                incices = [i for i in range(len(csv_data)) if i % 5 == 0]
                self.y = torch.tensor(csv_data[incices, -1])
                data = torch.tensor(csv_data[incices, : -1])
            else:
                incices = [i for i in range(len(csv_data))]
                data = torch.tensor(csv_data[incices])
            data = data[:,col]


            self.data = (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)

    def __getitem__(self, idx):
        if self.mode != "test":
            return self.data[idx].float(), self.y[idx].float()
        else:
            return self.data[idx].float()

    def __len__(self):
        return len(self.data)

all_feature = False
if all_feature:
    feature_dim = 93
else:
    feature_dim = 6



train_file = "covid.train.csv"
test_file = "covid.test.csv"
train_dataset = CovidDataset(train_file, "train",all_feature,feature_dim)
val_dataset = CovidDataset(train_file, "val",all_feature,feature_dim)
test_dataset = CovidDataset(test_file, "test",all_feature,feature_dim)

#  train_dataset 是处理后的数据
# for data in train_dataset:
#     print(data)  # data存放两个数据，分别是x, y

batch_size = 16
# 取一批数据打乱
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# --------------------------------------------------以上是获取数据部分


class MyModel(nn.Module):
    def __init__(self, inDim):
        super(MyModel, self).__init__()
        self.fcl = nn.Linear(inDim, 64)
        self.relul = nn.ReLU()
        self.fcl2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fcl(x)
        x = self.relul(x)
        x = self.fcl2(x)

        if len(x.size()) > 1:
            x = x.squeeze()
        return x


# model = MyModel(inDim=93)
#
# for batch_x, batch_y in train_loader:
#     # print(batch_x,batch_y)
#     predy = model(batch_x)

# ----------------------------------------------------------超参
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 初始化配置字典，包含学习率、训练轮数和动量参数
config = {
    "lr": 0.001,  # 学习率（Learning Rate），用于更新网络权重的步长
    "epochs": 20,  # 训练轮数（Epochs），数据集通过整个训练过程的次数
    "momentum": 0.9,  # 动量（Momentum），用于加速梯度下降过程，避免局部最小值
    "save_path": "model_save/best_model.path",
    "rel_path" : "pred.csv"
}


# -------------------------------------------------------------训练和验证
def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)
    plt_train_loss = []
    plt_val_loss = []

    min_val_loss = 9999999999999

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train()  # 模型调为训练模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_bat_loss = loss(pred, target,model)
            train_bat_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += train_bat_loss.cpu().item()

        plt_train_loss.append(train_loss / train_loader.__len__())

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target,model)
                val_loss += val_bat_loss.cpu().item()

            plt_val_loss.append(val_loss / val_loader.__len__())

            if val_loss < min_val_loss:
                torch.save(model, save_path)
                min_val_loss = val_loss

            print("[%03d/%03d] %2.2f sec(s) Trainloss: %.6f |Valloss: %.6f" % \
                  (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1]))

            plt.plot(plt_train_loss)
            plt.plot(plt_val_loss)
            plt.title("loss")
            plt.legend(["train", "val"])
            plt.show()


# ----------------------------------------------------------损失函数正则化，loss = loss + w*w
def mseLoss_with_teg(pred, target, model):
    loss = nn.MSELoss()
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)

    return loss(pred, target) + 0.00075 * regularization_loss








model = MyModel(inDim=feature_dim).to(device)
loss = mseLoss_with_teg
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])


#----------------------------------------------------验证
def evaluate(save_path, test_loader, device,rel_path):
    model = torch.load(save_path).to(device)
    rel = []
    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            rel.append(pred.cpu().item())

    with open(rel_path, "w",newline="") as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(["id","tested_positive"])
        for i in range(len(rel)):
            csvWriter.writerow([str(i), str(rel[i])])

        print("文件已经保存")






evaluate(config["save_path"],test_loader,device,config["rel_path"])