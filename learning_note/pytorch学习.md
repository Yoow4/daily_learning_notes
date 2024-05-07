# pytorch学习

1、GPU加速

2、自动求导

3、常用的网络层：轻松搭建网络

```
nn.Linear 
nn.Conv2d 
nn.LSTM 
nn.ReLU
nn.Sigmoid 
nn.Softmax
nn.CrossEntropyLoss
nn.MSE

```





## 回归问题

简单回归

逻辑回归，通过激活函数将数值回到1，可计算分类概率

### 手写数字识别MNIST



回归目标：one-hot编码，0~9变成十维向量

#### 加入非线性因素

ReLU(XW1+B1)

#### Inference:

$pred=W3*{W2(W1X+b1)+b2}+b3$

```python
argmax(perd)  #perd是一个十维向量
```



#### 实践

steps
①Load data②Build Model ③Train ④Test



## 张量数据类型

CPU tensor : torch.FloatTensor

GPU tensor :  torch.cuda.FloatTensor

注意这两种数据类型是不一样的，x.cuda0)会返回一个gpu上的引用，如：

```python
data=data.cuda( )

In [23]:isinstance(data, torch.cuda.DoubleTensor)
0ut[23]: True
```



pytorch没有内建string支持，如果需要的话，要用one-hot:[0,1,0,0...]或者Embedding：Word2vec、glove实现

### Dim0

torch.tensor(2.2)这样是一个维度为0的tensor

它的torch.size( [ ] )是空的但是，维度为0，是个标量

loss通常用这种tensor

### Dim1

Bias通常用这种tensor

Linear Input也用

### Dim2

Linear Input Batch用这种

### Dim3

RNN  Input Batch

适合nlp处理

### Dim4

CNN Input Batch

[b,c,h,w]

h：长，w：宽，c：通道，b：batch

### Mixed

```python
In [46]:a.shape
0ut[46]: torch.size([2，3，28,28])

In [47]: a.numel( )
0ut[47]:4704   #2*3*28*28

In [48]: a.dim( )
Out[48]: 4
```

numel是指tensor占用内存的数量



## 创建Tensor

```python
torch.from_numpy( )

torch.tensor([2.,3.2]) #直接从list创建，但不推荐这样
#推荐
torch.tensor(d1,d2,d3)#这种形式

torch.empty( ) #未初始化数据，后面记得给数据

```

### 设置默认类型

```python
In [74]:torch.tensor([1.2,3]).type( )
0ut[74]:'torch.FloatTensor‘   #默认是FloatTensor

torch.set_default_tensor_type(torch.DoubleTensor)
```

### 初始化

#### 随机初始化

```python
a=torch.rand(3,3)

torch.rand_like(a)

torch.randint(1,10,[3,3])
Out:
tensor([[8,4,2],
		[1,2,7],
		[3,6，2]])
		
torch.rand(3,3) # 正态分布N（0，1）
torch.normal(mean, std, size)

```

#### full

```
torch.full( )
```

#### arrange

```
 #生成一个等差序列的函数
torch.arrange()

sequence_tensor = torch.arange(10)
生成一个张量 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]。
sequence_tensor = torch.arange(1, 10, 2)
生成一个张量 [1, 3, 5, 7, 9]，起始值为 1，结束值为 10（不包含)
```

#### linspace

```python
#生成一维等间隔的数值序列
torch.linspace()

torch.linspace(start, end, steps=100, dtype=None, layout=torch.strided, device=None, requires_grad=False)

其中参数的含义为：

start：序列的起始值。
end：序列的结束值。
steps：生成的序列中的元素个数，默认为 100。
dtype：生成的张量的数据类型，默认为 None，即和输入的参数类型相同。
layout：生成的张量的布局，默认为 torch.strided。
device：生成的张量所在的设备，默认为 None，表示使用当前默认的设备。
requires_grad：是否需要梯度，默认为 False。

sequence_tensor = torch.linspace(0, 1, steps=10)
成一个张量 [0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0000]，包含了 10 个元素，起始值为 0，结束值为 1
```

#### ones/zeros/eye

```
torch.ones()
torch.zeros()
torch.eye()
```

#### randperm

```python
#生成一个随机排列的整数序列
torch.randperm( )

torch.randperm(n, dtype=None, layout=torch.strided, device=None, requires_grad=False)

n：生成随机排列的整数序列的长度。
dtype：生成的张量的数据类型，默认为 None，即使用默认的数据类型。
layout：生成的张量的布局，默认为 torch.strided。
device：生成的张量所在的设备，默认为 None，即使用当前默认的设备。
requires_grad：是否需要梯度，默认为 False。
常用于数据集的随机打乱和数据集的随机采样等场景。
```

## 索引与切片

### Indexing

#### select by steps

```python
In[145]:a[:,:,0:28:2,0:28:2].shape   #start: end: step
0ut[145]:torch.size([4,3,14，14])

#全部取也可以省略，直接用冒号
In [146]:a[:,:,::2,::2].shape
0ut[146]:torch.size([4,3，14，14])
```

#### select by specific index

 在张量 a 的第二个维度上（索引为 2 的维度，即图像的高度）选择索引为 0 到 7 的元素，并返回一个新的张量。

#### 符号...

```
selected_shape = a[..., 1].shape
```

在所有维度除了最后一个维度以外的所有维度上保持不变，并在最后一个维度上选择索引为 `1` 的所有元素的形状

...：表示剩余所有维度，该符号是为方便维度多的时候

#### select by mask

```python
#根据指定的掩码从输入张量中选择元素。
torch.masked_select(input, mask)

#.ge() 是用于比较的方法之一，其作用是检查一个对象是否大于或等于另一个对象。
object.ge(other)

In [170]:x=torch.randn(3,4)
tensor([[-1.3911，-0.7871,-1.6558，-0.2542]
[-0.9011, 0.5404，-0.6612, 0.3917],
[-0.3854，0.2968， 0.6040，1.5771]]
       
In[172]:mask =x.ge(0.5)
       tensor([[0,0,0,0],
               [0,1，0,0],
               [0,0,1，1]]，dtype=torch.uint8)
In [174]: torch.masked select(x, mask)
0ut[174]:tensor([0.5404，0.6040，1.5771])
       
In [175]: torch.masked_select(x,mask).shape
0ut[175]:torch.size([3])
```

#### select by flatten index

```python
#用于按照给定的索引从输入张量中获取元素
torch.take(input, indices)

相当于把元素做了flatten然后取元素

```

## 维度变换

###  view reshape

不变的是数据，变的是对数据的理解

```python
#torch.view() 是 PyTorch 中用于改变张量形状（维度）的方法之一。
result = input.view(*shape)
其中参数的含义为：

input：要改变形状的输入张量。
shape：一个包含要重新组织的维度大小的元组。这些维度大小的乘积应该与原始张量中的元素数量相同。

a=torch.rand(4,1,28,28)
#flatten
In[16]: a.view(4,28*28).shape
0ut[16]:torch.size([4，784]) 

#只关心行数据
In[17]:a.view(4*28，28).shape
out[17]: torch.size([112,28])

#只关注feature map 变成4张map
In[18]:a.view(4*1，28，28).shape
0ut[18]:torch.size([4,28，28])
In[19]:b=a.view(4,784)

数据的存储/维度顺序非常重要，需要时刻记住
```

### squeeze  v.s.  unsqueeze

####  unsqueeze

```python
#用于在指定位置增加一个维度的方法。它会在张量的指定位置（维度）上插入一个新的维度，从而改变张量的形状。 
result = torch.unsqueeze(input, dim)
input：要操作的输入张量。
dim：要插入新维度的位置（索引）。可以是一个标量值或一个元组。

#可以改变数据的理解方式，如
In[46]:a=torch.tensor([1.2,2.3])     #此处shape是[2]
In [47]: a.unsqueeze(-1)      #负数是在后面插入
out[47]:tensor([[1.2000],
				[2.3000]])			#此处shape是[2，1]
In [49]:a.unsqueeze(0)		 #正数是在前面插入
0ut[49]:tensor([[1.2000，2.3000]])	#此处shape是[1，2]
```

该函数可具体用在增加偏置bias的时候：

bias相当于给每个channel上的所有像素增加一个偏置

```
In[51]: b=torch.rand(32)
In[52]: f=torch.rand(4,32,14,14)
In[54]: b=b.unsqueeze(1).unsqueeze(2).unsqueeze(0)

In[55]: b.shape
0ut[55]: torch.size([1,32，1，1])
```

#### squeeze

`torch.squeeze()` 是 

```python
#PyTorch 中用于去除张量中尺寸为 1 的维度的方法。它会在张量中移除指定位置的维度，如果不指定位置，则会移除所有尺寸为 1 的维度。
result = torch.squeeze(input, dim=None)
 
In [60]: b.shape
out[60]:torch.size([1，32，1，1])

In [61]:b.squeeze().shape
0ut[61]: torch.size([32])

In [62]: b.squeeze(0).shape
0ut[62]:torch.size([32，1，1])
```

### Expand/repeat

Expand: broadcasting 只扩展不增加数据，推荐用这个，执行速度快且节省内存

Repeat: memory copied  实实在在地增加了数据

```
torch.expand() 是用于扩展张量的维度的方法，它允许你在指定位置扩展张量的维度，以满足某些计算的要求。但需要注意的是，torch.expand() 并不会增加张量的元素数量，而是使用现有的元素来填充新维度。

In [68]:a=torch.rand(4,32,14,14)
In [73]: b.shape
0ut[73]:torch.size([1,32,1，1])

In[70]:b.expand(4,32,14,14).shape
0ut[70]:torch.size([4，32，14，14])

In[72]:b.expand(-1,32,-1,-1).shape
Out[72]:torch.Size([1，32，1，1])
填-1就是不修改维度
```

repeat

```
torch.repeat() 是 PyTorch 中用于复制张量中的元素的方法。它允许你沿着指定的维度重复张量中的元素。这个方法会将张量中的元素在指定维度上复制若干次，从而扩大张量的尺寸。
result = torch.repeat(input, repeats)
input：要操作的输入张量。
repeats：一个包含了沿着每个维度要重复的次数的元组。如果 repeats 是一个整数，则所有维度都将重复相同的次数。

In [74]:b.shape
0ut[74]:torch.size([1,32,1,1])
In [75]:b.repeat(4,32,1,1).shape
0ut[75]:torch.size([4，1024，1，1])
```

### .t转置

**只适用于矩阵**

### Transpose

Tips:数据的维度顺序必须和存储顺序一致

```
用于交换张量维度的方法。它允许你在张量的维度之间进行转置操作，从而改变张量的形状。
result = torch.transpose(input, dim0, dim1)
input：要操作的输入张量。
dim0：要交换的维度之一。
dim1：要交换的维度之二。

交换[b c h w]为[b w h c]时还得用.contiguous( )保证数据连续
view会导致维度顺序关系变模糊，所以需要人为跟踪
```

### permute

```python
#用于按照指定顺序重新排列张量维度的方法。它允许你以一种更灵活的方式重新排列张量的维度，而不需要显式地提供维度的索引值。

result = torch.permute(input, *dims)
input：要操作的输入张量。
*dims：一个可变数量的参数，用于指定新的维度顺序。


In[98]:b.permute(0,2,3,1).shape
0ut[98]:torch.size([4，28，32，3])  #此时[b c h w]变为[b h c w]，如果涉及contiguous错误还得加.contiguous( )
```

## Broadcasting 自动维度扩张

