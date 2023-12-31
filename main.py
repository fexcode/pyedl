import numpy as np
import matplotlib.pyplot as plt
from time import time


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


class Tensor(object):
    def __init__(self, data, autograd: bool = False, creators: list = None, creation_op: str = None, id: int = None):
        """
        data: 数据
        autograd: 是否启用 autograd 机制
---------------------------以上为用户可以调用的-----------------------------
---------------------------以下为开发者要用的-------------------------------
        creators: 创建这个张量的张量,创建者        
        creation_op: 符号,这是一个API接口     \n 
        id: 每个张量都有一个唯一id \n

        self.grad  见   Tensor.backward        \n
        self.children:dict  :    {子节点id:接收到的梯度的个数}
            {id:cnt}
            这是一个计数器


        creation_op:        
            add  加法API
        """

        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}

        if (id is None):
            id = np.random.randint(0, 100000)
            # 如果没有id,就生成id
        self.id = id
        if (creators is not None):
            # 追踪一个张量有多少个子节点
            for c in creators:
                # c 为每一个创建者
                if (self.id not in c.children):
                    # 如果不在children里(第一次接受梯度),就记1次
                    c.children[self.id] = 1
                else:
                    # 如果在,就加一次
                    c.children[self.id] += 1

    def 一个张量是否已经从它在计算图中的所有孩子那里接受了梯度(self) -> bool:
        """
        用于内部计算，用户不需要调用。

        返回一个表示当前张量是否已经从它在计算图中的所有孩子那里接受了梯度的布尔值。

        返回:
        bool: 如果张量已经从所有孩子那里接受了梯度，则返回True；否则返回False。
        """
        for id, cnt in self.children.items():
            if (cnt != 0):
                return False
            return True

    def backward(self, grad=None, grad_origin=None):
        """
        递归向后传播，使用autograd机制。该方法能够将梯度传播回创建者张量，并在计算图中跟踪梯度的来源。

        参数：
        grad (Tensor): 反向传播过程中的梯度。
        grad_origin (Tensor): 梯度的来源。

        Raises:
        Exception: 如果尝试对同一个梯度来源多次进行反向传播。

        注意：
        1. 该方法仅在启用自动梯度计算（autograd）的情况下生效。
        2. 在调用该方法时，会根据梯度的来源和创建者，并根据对应的操作类型来执行相应的反向传播操作。
        """
        if self.autograd:
            if (grad_origin is not None):
                if (self.children[grad_origin.id] == 0):
                    # raise Exception("不能多次进行反向传播")
                    pass
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            创建者存在 = self.creators is not None
            梯度来源不存在 = grad_origin is None

            if (创建者存在 and (self.一个张量是否已经从它在计算图中的所有孩子那里接受了梯度() or 梯度来源不存在)):
                if (self.creation_op == "add"):
                    # print("100",self.grad.data.shape)
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if (self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())

                if (self.creation_op == "sub"):
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if (self.creation_op == "mul"):
                    new = self.grad*self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad*self.creators[0]
                    self.creators[1].backward(new, self)

                if (self.creation_op == "mm"):
                    act = self.creators[0]
                    weights = self.creators[1]
                    # print("121",act.data.shape,weights.data.shape)
                    # print("weights.transpose()",weights.transpose())
                    # print("!!!",self.grad.data.shape,weights.data.shape)
                    new = self.grad.mm(weights.transpose())
                    # print("-------new---------",new.data.shape)
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if (self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    # print("ones.data.shape",ones.data.shape)
                    self.creators[0].backward(self.grad*(self*(ones-self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(ones-(ones*self)))

                if "sum" in self.creation_op:
                    # 注意,传入的creation_op格式为sum_${dim}
                    # 我们要取dim
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    # print(dim,ds)
                    # print(self.creators[0].data.shape,self.grad.expand(dim, ds).data.shape)

                    self.creators[0].backward(self.grad.expand(dim, ds))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

    def sum(self, dim):
        """
        沿着指定的维度对数组进行求和操作

        参数:
        dim: 维度

        返回:
        Tensor：一个包含扩展数据的新Tensor对象。

        示例：
        如果我们有一个形状为(3, 4)的张量，调用expand(1, 2)将导致一个新的形状为(3, 8)的张量，其中原始数据在第二个维度上重复2次。

        """
        if (self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        """
        沿着指定维度扩展张量的数据，通过复制数据。

        参数：
        dim (int)：要沿其扩展张量数据的维度。
        copies (int)：在指定维度上要进行的数据复制次数。

        返回：
        Tensor：包含扩展数据的新Tensor对象。

        示例：
        如果原始张量的形状为(3, 4)，调用expand(1, 2)将会导致一个新的形状为(3, 8)的张量，其中原始数据在第二个维度上被复制2次。
        """
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape)+[copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    def transpose(self):
        """
        返回张量的转置。

        返回:
        Tensor: 表示当前张量转置后的结果。


        注意：
        1. 如果张量启用了自动梯度计算，在对张量执行转置操作时，会记录转置操作是由当前张量创建的。
        2. 返回的张量具有相同的值，但形状与当前张量的轴排列方式相反。
        """
        if (self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        """
        返回当前张量与另一个张量的矩阵乘法结果。

        参数:
        x (Tensor): 要与当前张量进行矩阵乘法的另一个张量。

        Returns:
        Tensor: 表示当前张量与输入张量进行矩阵乘法后的结果。

        注意：
        1. 如果张量启用了自动梯度计算，在进行矩阵乘法后会记录操作是由当前张量和输入张量创建的。
        2. 返回的张量包含当前张量与输入张量进行矩阵乘法后的结果。
        """
        if (self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        # print(self.data.shape,x.data.shape)
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1/(1+np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def __add__(self, other):
        if (self.autograd and other.autograd):
            # 加法的两个张量都支持 autograd
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add"
                          )
        return Tensor(self.data + other.data)

    def __neg__(self):
        if (self.autograd):
            return Tensor(self.data*(-1),
                          autograd=True,
                          creators=[self],
                          creation_op="neg")

        return Tensor(self.data*(-1))

    def __sub__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data-other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data-other.data)

    def __mul__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data*other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data*other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == '__main__':
    def 被忽略的测试():
        return False

    if 被忽略的测试():
        # 在0.1.0版本中,autograd机制被修复,接口更改,所以请忽略这两个测试
        if "测试1" == "0.0.0":
            print("-----------测试1,Tensor的可信性   0.0.0------------")
            a = Tensor([1, 1, 4, 5, 1, 4])
            print(a)
            print(a+Tensor(1))
        if "测试2" != "0.0.1":
            print("----------测试加法autograd的可行性  0.0.1----------")
            a = Tensor([1, 1, 4, 5, 1, 4])
            b = Tensor([2, 2, 2, 2, 2, 2])
            c = a+b
            c.backward(Tensor(np.array([1, 1, 1, 1, 1])))
            print(a.grad, b.grad, c.creators)
    if "测试3" == "0.1.0  and  0.1.1":
        print("-----------autograd 修复后多节点反向传播测试-----------")
        a = Tensor([1, 1, 4, 5, 1], autograd=True)
        b = Tensor([6, 5, 4, 5, 1], autograd=True)
        c = Tensor([9, 8, 7, 5, 1], autograd=True)
        d = a + b
        e = b + c
        f = d + e

        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print(b.grad.data == np.array([2, 2, 2, 2, 2]))
        print('-----------0.1.1测试-----------------')
        d = a+(-c)
        d.backward(Tensor(np.array([2, 2, 2, 2, 2])))
        print(d.grad, c.grad)

    if "测试4" == "0.1.5":

        print(
            "--------------------autograd终于TM写完了,来测试一下,老天保佑不要有BUG-------------------------")
        print("-------------------------训练一个神经网络---------------------------")

        np.random.seed(0)

        data = Tensor(np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]]), autograd=True)
        target = Tensor(np.array([
            [0],
            [1],
            [0],
            [1]
        ]), autograd=True)

        w = list()
        w.append(Tensor(np.random.rand(2, 3), autograd=True))
        w.append(Tensor(np.random.rand(3, 1), autograd=True))

        lossls = []  # 损失列表,画图要用

        print("""
              ----------------
            |    全连接(2,3)     |
            |    全连接(3,1)     |
              ----------------
        """)

        print("training...")

        EPOCH = 200

        startTime = time()
        for i in range(EPOCH):
            pred = data.mm(w[0].mm(w[1]))
            # 预测
            loss = ((pred-target)*(pred-target)).sum(0)
            # 我为什么不写成      (pred-target)**2   ?
            # 因为我没有定义平方运算的autograd
            # 这是损失函数,以后会专门用一个类储存
            loss.backward(Tensor(np.ones_like(loss.data)))
            # AUTOGRAD!!!
            # 学习
            # 曾经最难的向后传播现在...,太优雅了
            for w_ in w:
                assert isinstance(w_, Tensor)
                test = w_.grad
                w_.data -= w_.grad.data*0.1
                # alpha
                w_.grad.data *= 0
                # 梯度清零
            lossls.append(loss.data[0])
        endTime = time()
        t = (endTime-startTime)
        print(f"训练用时{t}")

        print(w)

        x = range(len(lossls))
        # 绘制折线图
        plt.plot(x, lossls, marker='*')
        # 添加标题和标签
        plt.title(
            f'只要我不说,你就不知道我调试了多长时间,{t*100}ms训练{EPOCH}轮', fontproperties='KaiTi', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # 显示图表
        plt.show()
