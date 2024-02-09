# 神经网络文档

## 神经元层

### Layer 类

`Layer` 类是基本的神经网络层。

#### 方法:

- `__init__()`: 初始化参数的构造方法。
- `get_parameters()`: 获取层的参数的方法。

### Linear 类

Linear 类实现线性变换。

#### 方法:

- `__init__(n_inputs, n_outputs)`: 构造方法，输入特征数量为 n_inputs，输出特征数量为 n_outputs。
- `forward(X)`: 前向传播方法，计算线性变换的输出。

## 损失函数

### MSELoss 类

`MSELoss` 类实现均方误差损失函数。

#### 方法:

- `__init__()`: 初始化方法。
- `forward(pred, target)`: 计算预测值和目标值之间的均方误差。

## 非线性层

### Tanh 类

`Tanh` 类实现双曲正切非线性函数。

#### 方法:

- `__init__()`: 初始化方法。
- `forward(input)`: 对输入进行双曲正切转换。

### Sigmoid 类

`Sigmoid` 类实现 sigmoid 非线性函数。

#### 方法:

- `__init__()`: 初始化方法。
- `forward(input)`: 对输入进行 sigmoid 转换。
