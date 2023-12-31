# 简单入门 Pyedl 的模块————pyedl.head

> pyedl 是模块化的  
> 所以,你如果想入门一个神经网络,可能需要 import 一大堆东西<br>
> 比如:
>
> ```python
> from pyedl.main import Tensor
> from pyedl.nn import Sequential, Linear, MSELoss, Sigmoid, Tanh
> from pyedl.optim import SGD
> from matplotlib import pyplot as plt
> import numpy as np
> ```
>
> 这会劝退很多新手  
> 这个模块集成了这些,只需要 `from head import *` 就行了<br>  
> 我知道我写这个库会被很多大佬喷  
> 但是只要新手觉得简单,有了入门的兴趣  
> 我的脸皮还可以再厚一点(笑)

## 集成了什么?

```python
from pyedl.main import Tensor
from pyedl.nn import Sequential, Linear, MSELoss, Sigmoid, Tanh
from pyedl.optim import SGD
from matplotlib import pyplot as plt
import numpy as np
```

## 函数

### drawLossLs

#### 函数名称：

- drawLossLs

#### 函数功能：

- 绘制损失列表的折线图

#### 参数列表：

- lossList: list，损失列表
- title: str，图表标题，默认为"PYEDL!!!"

#### 返回值：

- 无

#### 示例：

```python
    lossList = [0.5, 0.4, 0.3, 0.2, 0.1]
    drawLossLs(lossList, "训练损失变化图")
```
---
### easyTrain

#### 函数名称：

- easyTrain

#### 函数功能：

- 简单训练模型并返回损失列表

#### 参数列表：

- data: 输入数据
- target: 目标数据
- model: 模型
- criterion: 损失函数，默认为MSELoss()
- optim: 优化器，默认为None
- epoch: 训练轮数，默认为10
- returnLossList: 是否返回损失列表，默认为False
- printLoss: 是否打印损失，默认为False

#### 返回值：

- 如果returnLossList为True，则返回损失列表，否则返回None

#### 示例：

```python
data = [1, 2, 3, 4, 5]
target = [0.5, 1.5, 2.5, 3.5, 4.5]
model = NeuralNetwork()
criterion = MSELoss()
loss = easyTrain(data, target, model, criterion, epoch=20, returnLossList=True, printLoss=True)

