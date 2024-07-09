# MMdet笔记

参考链接：[5 MMDetection 代码教学_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Av4y1475i/?spm_id_from=333.337.search-card.all.click&vd_source=bf7b9535de982f1d288138463991a3f7)

# configs配置文件

## 深度学习模型的训练涉及

模型结构				模型层数、通道数等等

数据集					用什么数据训练模型：数据集划分、数据文件路径、数据增强策略等等

训练策略				梯度下降算法、学习率参数、batch_size、训练总轮次、学习率变化策略等等	

运行时					GPU、分布式环境配置等等

一些辅助功能		如打印日志、定时保存checkpoint等等

## 配置文件中

定义了一个完整的训练过程

`model`字段定义模型

`data`字段定义数据

`optimizer`、`lr_config`等字段定义训练策略

`load_from`字段定义与训练模型的参数文件





# mmdet核心工具包





# 训练自己的检测模型

## 基于微调训练

- 使用基于COCO预训练的检测模型作为梯度下降的“起点”
- 使用自己的数据进行“微调训练”，通常需要降低学习率

具体到MMDetection

- 选择一个基础模型，下载对应的配置文件和预训练模型的参数文件
- 将数据整理成MMDetection支持的格式，如COCO格式或者自定义格式
- 修改配置文件（可以通过继承的方式，不用把一整个配置文件贴过来再一条一条改）
  - 修改配置文件中的数据路径
  - 修改模型的分类头
  - 设置加载预训练模型
  - 修改优化器配置（学习率、训练轮次等）
  - 修改一些杂项

## 配置文件.py修改

### model修改

默认coco是分80类，需要几类就在`num_classes`改成几 

### data修改

samples_per_gpu 调整batch_size防止爆显存，可以用数据量看看多少整除合适



`type` 数据集类型

`ann_file`标注文件，微调训练时需要更改该文件

`img_prefix`存放所有图片的路径，微调时也需要改

`pipeline`定义数据加载过程，不需要改，该部分主要是定义如何对数据进行处理，如裁剪缩放、水平翻转、像素值归一、转换数据类型为tensor等

`classes`如果更改了分类数量要修改



具体操作：

新建一个.py文件

```
_base_ = ['要继承的配置文件.py']
```

保持层级不变，把要改的数据改了就行

涉及到目录时，最好边填边用ls检查是否填对了



### load_from参数加载

`load_from`加载预训练模型参数 

### 训练轮次

`runner` 修改`max_epochs`

### 学习率

`optimizer`修改lr

`lr_config`决定学习策略，一般从头训练才用得上，微调可以不管

### 输出日志

` log_config`修改interval越小打印得越快



# 观察结果

利用代码

```python
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = "config.py"
checkpoint_file = "checkpoint.pth"
img = "image.jpg"

model = init_detector(config_file, checkpoint_file)
result = inference_detector(model, img)
show_result_pyplot(model,img,result)
```

或用命令

```
mim test mmdet config.py --checkpoint /path/to/checkpoint.pth --show-dir /path/to/save/
```



# COCO数据集标注格式

所有标注信息存储在一个JSON对象中

包含以下字段

```
{
	"image"
	"annotations"
	"categories"
	"info"
	"licenses"这俩不用管
}
```

 

YOLOv3模型

主干网络(backbone)——>颈部——>检测头



mmdetection对模型都在配置文件里进行了模块分割



# Runner

包括了下面的所有东西

```
model
# 训练所用数据
    train_dataloader
# 训练相关配置
    train_cfg
# 优化器封装，MMEngine 中的新概念，提供更丰富的优化选择。
    # 通常使用默认即可，可缺省。有特殊需求可查阅文档更换，如
    # 'AmpOptimWrapper' 开启混合精度训练
    optim_wrapper
# 参数调度器，用于在训练中调整学习率/动量等参数
   param_scheduler
# 验证所用数据
    val_dataloader
# 验证相关配置，通常为空即可
    val_cfg
# 加载权重的路径 (None 表示不加载)
    load_from=None,
# 从加载的权重文件中恢复训练
    resume=False
```



使用配置文件时，你通常不需要手动注册所有模块。例如，`torch.optim` 中的所有优化器（如 `Adam` `SGD`等）都已经在 `mmengine.optim` 中注册完成。使用时的经验法则是：尝试直接使用 `PyTorch` 中的组件，只有当出现报错时再手动注册。

# Model

模型基类BaseModel 





[**forward**](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.model.BaseModel.html#mmengine.model.BaseModel.forward): `forward` 的入参需通常需要和 [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的输出保持一致 (自定义[数据预处理器](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/model.html#datapreprocessor)除外)，如果 `DataLoader` 返回元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的解包后的参数；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 解包后的参数。

在Python中，`data`、`*data`和`**data`是用来处理函数参数的不同方式，尤其在处理不定数量的参数或者关键字参数时非常有用。这些概念在深度学习框架（如PyTorch）中的模型定义和数据加载时尤为重要，因为它们允许模型的`forward`方法灵活地处理不同形式的输入数据。下面是对这三种形式的解释：



1. `data`

   :

   - 当我们谈到`data`时，我们指的是变量本身所持有的数据。这个变量可以是任何类型的数据，比如数字、字符串、列表、元组、字典等。
   - 在深度学习中，如果你的`DataLoader`返回一个元组或字典作为数据批次，那么这个元组或字典整体就是`data`。

2. `*data`

   :

   - `*data`用于将元组（tuple）类型的数据进行解包（unpack）。这意味着如果你的函数期望接收多个独立的参数，而你有一个元组包含了所有这些参数，你可以通过在元组变量前加`*`来将其解包为独立的参数。
   - 例如，如果你的`DataLoader`返回一个元组`(x, y)`，并且你的模型的`forward`方法定义为接收两个参数，如`forward(self, x, y)`，那么你可以在调用时使用`*data`来将元组解包为两个独立的参数。

3. `**data`

   :

   - `**data`用于将字典（dict）类型的数据进行解包。这允许你将一个字典解包为关键字参数（key-value pairs），如果函数期望接收具有特定名称的关键字参数，这种方式非常有用。
   - 例如，如果你的`DataLoader`返回一个字典，如`{'x': value1, 'y': value2}`，并且你的模型的`forward`方法定义为接收两个关键字参数，如`forward(self, x, y)`，那么你可以在调用时使用`**data`来将字典解包为关键字参数。
      总结来说，`data`是原始数据，`*data`用于从元组中解包位置参数，而`**data`用于从字典中解包关键字参数。这些技巧在处理不同类型和数量的输入数据时提供了极大的灵活性和便利。

## 数据预处理器(DataPreprocessor)

在BaseModel初始化时会构造一个默认的BaseDatePreprocessor，它负责将数据搬运到指定设备



该类可完成数据搬运、归一化、数据增强等功能

如果Dataloder输出和模型输入类型不匹配，最好是进行预处理，在不破坏模型和数据已有接口的情况下完成适配



# DATASET、DATALOADER

- `train_dataloader`：在 `Runner.train()` 中被使用，为模型提供训练数据
- `val_dataloader`：在 `Runner.val()` 中被使用，也会在 `Runner.train()` 中每间隔一段时间被使用，用于模型的验证评测
- `test_dataloader`：在 `Runner.test()` 中被使用，用于模型的测试



## 自定义数据集

像使用 PyTorch 一样，自由地定义自己的数据集，或将之前 PyTorch 项目中的数据集拷贝过来。如果你想要了解如何自定义数据集，可以参考 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

使用MMEngine的数据集基类

# EVALUATION