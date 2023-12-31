from pyedl.main import Tensor
from pyedl.nn import Sequential, Linear, MSELoss, Sigmoid, Tanh
from pyedl.optim import SGD
from matplotlib import pyplot as plt
import numpy as np


def drawLossLs(lossList: list, title="PYEDL!!!"):
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


def easyTrain(data, target, model, criterion=MSELoss(), optim=None, epoch=10, returnLossList=False, printLoss=False):
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
        lossList.append(loss.data[0])
    if returnLossList:
        return lossList
