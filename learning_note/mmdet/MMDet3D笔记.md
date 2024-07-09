# MMDet3D笔记

MMCV基础库

MMDet检测

MMSeg

MMEngine V1.1.x新架构



# 源码安装

```
#安装min包管理工具
pip install openmin
#从源码安装
git clone https://github.com/open-mmlab/mmdetection3d.git
cd  mdet3d
#(可选)切换分支
min install -e .
```

安装完后可利用验证

```
import mmdet3d
mmdet3d.__version__
```

观察版本是否正确输出



# 使用预训练模型推理

1、去GitHub的算法页面、或使用mim工具下载预训练模型

2、准备数据（点云、图像、标注文件）

3、调用Python API或使用demo程序实现推理、以及结果可视化



## demo文件夹

pcd_demo.py

调用LidarDet3DInferencer()





## 如何写相机.json文件

```
{
	"images":[
		{
			"file_name":"xxx.png",
			"id":0,
			"camera_intrinsic":[
				[
					1000,	#焦距
					0.0,
					683.0	#相机中心位置，通常是图片的长宽大小除以2
				],
				[
					0.0,
					1000,	#焦距
					384.0,	#相机中心位置，通常是图片的长宽大小除以2
				]，
				[
					0.0,
					0.0,
					1.0		#齐次坐标
				]
			],
			"width":1366,
			"height":768
		}
	]
}
```



# 点云可视化

安装Open3D

```
pip install open3d
```

MMDetection 3D基于Open3D构建了若干可视化点云工具

可视化时需要雷达坐标系中的坐标



使用MMDetection3D进行训练





# 3D目标检测算法

参考：[清华大学导师亲授！3D目标检测超硬核课程新鲜出炉~_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Sc411K7L1/?spm_id_from=333.337.search-card.all.click&vd_source=bf7b9535de982f1d288138463991a3f7)

## 基于点云数据的3D检测算法

### 基本思路1：基于体素（Voxel）

空间3维，加上特征通道维度是4D的Tensor

做法：

1. 在空间中划分格子
2. 基于格子内的点计算对应格子的特征，得到3D特征体
3. 送入3D卷积网络产生特征图，再基于检测头产生预测结果

如：VoxelNet、SECOND等

### 基本思路2：基于点云的鸟瞰图

做法：与体素的方法类似，但只在地面上划分格子，高度维不划分，可以将点云直接转化为2D特征图，之后基于2D检测算法产生预测框，由于检测目标有所区别，需要更改回归分支。

如：Pixor、Complex-YOLO、PointPillars、CenterPoint等

### 基本思路3：基于点

做法：使用PointNet++，直接基于非结构化的点云产生特征，再基于点的特征产生预测框

如：PointRCNN等

### VoxelNet(2017)&SECOND(2018)

流程：

1. 将空间划分为体素，在体素内使用VFE(Voxel Feature Encoding)网络提取局部点云特征，得到三维的特征体
2. 将特征送入3D卷积网络，进一步提高表达能力，将最终输出在竖直方向压缩，得到2D特征图
3. 将2D特征图送入RPN网络产生3D框预测



VFE(Voxel Feature Encoding):用一层全连接层(含BN和ReLU)对每个点的原始坐标进行变换，再最大池化得到全局特征，最后将全局特征拼接到逐点特征上作为输出



问题：较多3D卷积层，速度慢；角度回归精度不够

SECONDA解决：

稀疏Tensor:只对有点的地方进行计算

目标值改为角度正弦值+二分类区分0和180°

### PointPillars(2019)

只在地面上划分二维网格，不在高度方向划分格子，形成一系列柱体(Pillars)

每个柱体内使用简化版的PointNet编码点云特征，得到2D特征图

完全舍弃3D卷积，速度快精度高

### CenterNet(2019)

传统：以框表示物体

核心思想：以中心点表示物体

流程：

 由图像通过主干网络降采样得到特征图

逐点回归热力图

以及降采样造成的局部偏移量

或回归其他属性，如3D属性

### CenterPoint(2020)

原始点云体素化后送入3D卷积网络，再在竖直方向堆叠，得到2D特征图

用CenterNet检测头来预测关键点，边界框和转角

### Single-stride Sparse Transformer(2021)

核心思路：利用Transformer的attention机制，能获取足够的感受野且适用于稀疏的点云体素



## 目标检测

直接基于PointNet++的分割模型，产生逐点的特征，再基于点特征产生预测框



如果换成抓取位姿，就是在获得点的特征后，预测抓取位姿

## PointnetRCNN（2018）



# 基于多模态的3D检测算法



基于纯视觉的3D检测算法

### 基本思路1：伪点云

基于单目或多目图像预测每个位置的深度，利用点云检测算法

### 基本思路2：单目视觉

基于2D检测的算法框架，增加额外的回归分支以预测3D框的全部7个属性

如：等

### 基本思路3：多视角融合

基于多目相机获得对场景更好的理解，不同视角之间的关联可以基于Attention机制学习

如：等



# Pytorch模型部署基础知识

参考：[1. PyTorch 模型部署基础知识_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Tx4y1F768/?spm_id_from=333.788&vd_source=bf7b9535de982f1d288138463991a3f7)

## ONNX模型结构

ONNX(Open Neural Network Exchange)

一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。不同的训练框架可采用相同格式存储模型并交互。

```
netron.app网页
把生成的onnx模型拖进去就能看到结构以及输入输出信息
```

https://onnx.ai/onnx/operators可以查看具体算子的属性

```
model = LeNet()
model.eval()#进入evaluation模型，去掉Dropout，让参数固定住
x = torch.rand((和输入一样大小),device='cuda:0')
torch.onnx.export(model,(x,False),f='resnet18.onnx',input_names=['input'],output_names=['output'],opset_version=11)
```

torch.onnx.export参数介绍

- model : pytorch模型，可以是torch.nn.module对象

- args：Tensor,tuple。必须保证model(args)能成功运行

- f : 存储onnx模型的文件对象

- input_names：输入节点的名称列表

- output_names：输出节点的名称列表

- opset_version：opset版本号

- ```
  dynamic_axes=dict(
  	input=({0:'batch'}),
  	output=({0:'batch'}))
  ```

  

## 使用推理引擎TensorRT对ONNX模型进行推理

Build Phase ：对ONNX模型转化和优化，输出优化后模型

利用trtexec

```
trtexec --onnx=模型所在路径 --saveEngine=保存名字.plan
```



# MMDeploy

参考：[2. 安装配置 MMDeploy 环境_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1yX4y1X7jp/?spm_id_from=333.788)

包含:

模型转换器：ONNX、TorchScript

推理SDK:C/C++、Python、C#、Java