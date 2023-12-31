# 优化器
## SGD优化器

### 简介
SGD（Stochastic Gradient Descent）是一种常用的优化算法，用于机器学习和深度学习模型的训练。本文档描述了一个简单的SGD优化器的类实现，包括初始化参数、梯度归零和参数更新功能。

### 类定义

```python
class SGD():
    def __init__(self, parameters, alpha=0.1):
        '''
        构造函数，初始化SGD优化器。

        参数：
        - parameters：要优化的参数
        - alpha：学习率，默认为0.1
        '''

    def zero(self):
        '''
        将梯度置零
        '''

    def step(self, zero: bool = True):
        '''
        根据当前梯度更新参数

        参数：
        - zero：一个布尔值，指定是否将梯度归零。默认为True。
        '''
```
### 使用示例
```python
# 创建一个SGD优化器实例
optimizer = SGD(model.parameters(), alpha=0.01)

# 在训练循环中使用优化器
for input, target in dataset:
    # 1. 清除过往梯度
    optimizer.zero_grad()
    
    # 2. 前向传播
    output = model(input)
    
    # 3. 计算损失
    loss = loss_function(output, target)
    
    # 4. 反向传播
    loss.backward()
    
    # 5. 更新模型参数
    optimizer.step()
```