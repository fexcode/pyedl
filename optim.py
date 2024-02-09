class SGD:
    def __init__(self, parameters, alpha=0.1):
        """
        初始化SGD优化器。
        随机梯度下降

        参数：
        parameters：要优化的参数
        alpha：学习率，默认为0.1
        """
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        """
        将梯度置零
        """
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        """
        根据当前梯度更新参数

        参数：
        zero：一个布尔值，指定是否将梯度归零。默认为True。
        """
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha
            if zero:
                p.grad.data *= 0
