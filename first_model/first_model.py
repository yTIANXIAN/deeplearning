import torch
import matplotlib.pyplot as plt
import random

# 生成模拟数据
def create_data(w, b, data_num):
    """
    生成线性模型数据，用于模拟真实世界中的数据分布。

    参数:
    w (torch.Tensor): 真实模型的权重。
    b (torch.Tensor): 真实模型的偏置。
    data_num (int): 生成的数据量。

    返回:
    tuple: 包含生成的特征数据x和标签数据y。
    """
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b
    noise = torch.normal(0, 0.001, y.shape)

    y += noise
    return x, y

# 设置数据集大小和真实参数值
num = 500
true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor(1.1)
X, Y = create_data(true_w, true_b, num)

# 数据提供器，用于批量读取数据
def data_provider(data, labels, batch_size):
    """
    生成器函数，用于按批次提供数据。

    参数:
    data (torch.Tensor): 特征数据。
    labels (torch.Tensor): 标签数据。
    batch_size (int): 每批次的数据量。

    生成:
    tuple: 包含数据批次和对应标签批次的元组。
    """
    length = len(data)
    indices = list(range(length))
    random.shuffle(indices)

    for each in range(0, length, batch_size):
        get_indices = indices[each:each + batch_size]
        get_data = data[get_indices]
        get_label = labels[get_indices]
        yield get_data, get_label

# 定义线性模型
def fun(x, w, b):
    """
    线性模型函数，用于预测输出。

    参数:
    x (torch.Tensor): 输入特征。
    w (torch.Tensor): 权重参数。
    b (torch.Tensor): 偏置参数。

    返回:
    torch.Tensor: 预测的输出值。
    """
    pred_y = torch.matmul(x, w) + b
    return pred_y

# 计算平均绝对误差损失
def maeLoss(y_pred, y):
    """
    计算预测值和真实值之间的平均绝对误差。

    参数:
    y_pred (torch.Tensor): 预测的输出值。
    y (torch.Tensor): 真实的输出值。

    返回:
    torch.Tensor: 平均绝对误差。
    """
    return torch.sum(abs(y_pred - y)) / len(y)

# 随机梯度下降优化器
def sgd(paras, lr):
    """
    随机梯度下降优化算法，用于更新模型参数。

    参数:
    paras (list): 需要更新的模型参数列表。
    lr (float): 学习率。
    """
    with torch.no_grad():
        for para in paras:
            para -= para.grad * lr
            para.grad.zero_()

# 初始化模型参数和学习率
lr = 0.009
w_0 = torch.normal(0, 0.01, true_w.shape, requires_grad=True)
b_0 = torch.tensor(0.01, requires_grad=True)

# 训练模型
epochs = 49
# 迭代训练模型，epochs为训练的轮数
for epoch in range(epochs):
    data_loss = 0
    # 使用data_provider函数生成小批量数据进行迭代训练，批量大小为16
    for x, y in data_provider(X, Y, 16):
        # 使用当前参数w_0和b_0对输入x进行预测
        pred_y = fun(x, w_0, b_0)
        # 计算预测值与真实值y之间的MAE损失
        loss = maeLoss(pred_y, y)
        # 反向传播计算损失关于参数的梯度
        loss.backward()
        # 使用SGD优化算法更新参数w_0和b_0，lr为学习率
        sgd([w_0, b_0], lr)
        # 打印当前轮次和损失值
        print("epoch:", epoch, "loss:", loss)


# 打印真实和训练得到的模型参数
idx = 0
print("真实的函数值", true_w, true_b)
print("训练得到的函数值", w_0, b_0)

# 绘制数据和模型预测结果
plt.plot(torch.detach(X[:, idx]), torch.detach(X[:,idx]*w_0[idx]+b_0))
plt.scatter(torch.detach(X[:, idx]), torch.detach(Y),1)
plt.show()
