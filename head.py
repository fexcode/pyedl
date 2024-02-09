from pyedl.main import Tensor
from pyedl.nn import Sequential, Linear, MSELoss, Sigmoid, Tanh, CrossEntropyLoss
from pyedl.optim import SGD
from matplotlib import pyplot as plt
import numpy as np


def drawLossLs(lossList: list, title="PYEDL!!!"):
    """
    函数名称：drawLossLs

    函数功能：绘制损失列表的折线图

    参数列表：
    lossList: list，损失列表
    title: str，图表标题，默认为"PYEDL!!!"

    返回值：无

    示例：
    lossList = [0.5, 0.4, 0.3, 0.2, 0.1]
    drawLossLs(lossList, "训练损失变化图")

    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    x = range(len(lossList))
    # 绘制折线图
    plt.plot(x, lossList, marker='*')
    # 添加标题和标签
    plt.title(
        title, fontproperties='KaiTi', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # 显示图表
    plt.show()


def easyTrain(data, target, model, criterion=MSELoss(), optim=None, epoch=10, returnLossList=True, printLoss=False):
    """
    函数名称：easyTrain

    函数功能：简单训练模型并返回损失列表

    参数列表：

    data: 输入数据
    target: 目标数据
    model: 模型
    criterion: 损失函数，默认为MSELoss()
    optim: 优化器，默认为None
    epoch: 训练轮数，默认为10
    returnLossList: 是否返回损失列表，默认为False
    printLoss: 是否打印损失，默认为False
    返回值：如果returnLossList为True，则返回损失列表，否则返回None

    示例：
    data = [1, 2, 3, 4, 5]
    target = [0.5, 1.5, 2.5, 3.5, 4.5]
    model = NeuralNetwork()
    criterion = MSELoss()
    loss = easyTrain(data, target, model, criterion, epoch=20, returnLossList=True, printLoss=True)
    """
    lossList = []
    if optim is None:
        optim = SGD(parameters=model.get_parameters(), alpha=0.01)
    print("============================Training================================")
    for i in range(epoch):
        pred = model.forward(data)  # 预测
        loss = criterion.forward(pred, target)  # 比较
        loss.backward(Tensor(np.ones_like(loss.data)))  # 学习
        optim.step()
        if printLoss:
            print(loss)
        if criterion is MSELoss():
            lossList.append(loss.data[0])
        elif criterion is CrossEntropyLoss():
            lossList.append(loss.data)
        else:
            lossList.append(loss.data)
    if returnLossList:
        return lossList
