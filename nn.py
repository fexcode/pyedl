import numpy as np
from pyedl.main import Tensor
from pyedl.optim import SGD

############################### 神经元层 #################################


class Layer():

    """
    基本的神经网络层类
    """

    def __init__(self) -> None:
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    '''
    Linear

    功能: 实现线性变换

    输入:
    n_inputs: int, 输入特征的数量
    n_outputs: int, 输出特征的数量

    输出:
    无

    参数:
    weight: Tensor, 权重矩阵
    bias: Tensor, 偏置向量

    示例:
    linear = Linear(3, 2)
    input = Tensor(np.array([[1, 2, 3]]))
    output = linear.forward(input)

    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        W = np.random.randn(input_size, output_size)*np.sqrt(2.0/(input_size))
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(output_size), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, X):
        return X.mm(self.weight) + self.bias.expand(0, len(X.data))


class Sequential(Layer):
    def __init__(self, layers=[]):
        super().__init__()
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params

############################### 损失函数 #################################


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)

############################### 非线性层 #################################


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


if __name__ == "__main__":
    if "测试5" != "0.1.8 & 0.1.12":
        print("--------------------基本神经网络层,可行性测试------------------------")
        np.random.seed(0)

        data = Tensor(
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        modle = Sequential([Linear(2, 3),Sigmoid(), Linear(3, 1),Sigmoid()])

        optim = SGD(parameters=modle.get_parameters(), alpha=0.01)
        criterion = MSELoss()

        EPOCH = 100
        for i in range(EPOCH):
            pred = modle.forward(data)
            loss = criterion.forward(pred, target)
            loss.backward(Tensor(np.ones_like(loss.data)))
            optim.step(zero=True)

            print(loss)
        print(modle.forward(data))
