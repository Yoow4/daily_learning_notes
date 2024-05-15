# Graspnet-baseline复现

## 直接按照README.md安装

创建环境

```
conda create -n hyw_graspnet_baseline_py38 python=3.8
```

运行

```
cd pointnet2
python setup.py install
```

报错1：ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch)) ValueError: Unknown CUDA arch (8.6) or GPU not supported



```
export TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0"
```







报错2：

ImportError: cannot import name 'NDArray' from 'numpy.typing' (/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline_py38/lib/python3.8/site-packages/numpy/typing/__init__.py)

升级numpy

提示不兼容：

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
graspnetapi 1.2.11 requires numpy==1.20.3, but you have numpy 1.24.4 which is incompatible.

不管它



报错3：AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.

```
_MAX_FLOAT = np.maximum_sctype(np.float)
_FLOAT_EPS = np.finfo(np.float).eps
```

修改为：

```
_MAX_FLOAT = np.maximum_sctype(np.float64)
_FLOAT_EPS = np.finfo(np.float64).eps
```



下载checkpoint

修改command_demo.sh

```
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/log_kn/checkpoint-rs.tar
```



报错4：

RuntimeError: CUDA error: no kernel image is available for execution on the device



这个警告表明你的 PyTorch 安装不支持你当前使用的 NVIDIA GeForce RTX 3090 Ti GPU 的 CUDA 架构版本（sm_86）。目前的 PyTorch 安装只支持 CUDA 架构版本为 sm_37、sm_50、sm_60、sm_70 和 sm_75 的 GPU。





# 修改requirements.txt中torch==1.11.0编译

报错：ImportError: cannot import name 'NDArray' from 'numpy.typing' (/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline_py38/lib/python3.8/site-packages/numpy/typing/__init__.py)

```
pip install -U numpy
```

升级numpy

提示不兼容：

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
graspnetapi 1.2.11 requires numpy==1.20.3, but you have numpy 1.24.4 which is incompatible.

运行命令

```
sh command_demo.sh 
```

报错：AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.

将/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline/lib/python3.9/site-packages/transforms3d/quaternions.py代码中的 `np.float`：

```
x = np.float(5.0)
```

替换为：

```
x = float(5.0)
```

这样就可以避免这个错误了。



再次运行，报错：

ImportError: cannot import name 'container_abcs' from 'torch._six' (/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline_py38/lib/python3.8/site-packages/torch/_six.py)



报错2：ImportError: Could not import _ext module.
Please see the setup instructions in the README: https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst



曾经认为：pytorch版本和CUDA环境必须匹配，配置以前的代码需要修改成显卡的CUDA版本对应的pytorch



# 尝试在高版本CUDA下使用低版本的pytorch，结论：不可行；



更新想法：可以装以前版本的pytroch

hyw_graspnet_baseline_py38_torch16



#### ValueError：Unknown CUDA arch（8.6）or GPU not supported。

参考链接：[FastBEV复现 Ubuntu_valueerror: unknown cuda arch (8.6) or gpu not sup-CSDN博客](https://blog.csdn.net/Furtherisxgi/article/details/129758236)

安装pytorch=1.6.0,

在RTX3090ti上编译的时候遇到了这个问题：ValueError：Unknown CUDA arch（8.6）or GPU not supported。解决方法：将conda环境所在文件夹中的cpp_extension.py内容从：

```
named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere', '8.0+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0']
```

修改为：

```
named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere', '8.0+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0','8.6']
```

然后这个问题就解决啦！





#### conda 虚拟环境python的sys.path包含了~/.local/lib

参考链接：[(12 条消息) conda 虚拟环境python的sys.path包含了~/.local/lib，如何解决？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/458876195)



解决办法：修改虚拟python环境下site.py中的USER_SITE的路径

进入虚拟环境，输入python -m site 可以查看默认的库路径

输入python -m site -help 可以查看site.py的路径，

```
/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline_py38_torch16/lib/python3.8/site.py
```

修改USER_SITE的值为该虚拟环境的库的路径

我修改的值为

```
USER_SITE = '/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline_py38_torch16/lib/python3.8/site-packages'
```





# conda虚拟环境复现流程pyt1.6

下载

```
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline/
```

创建虚拟环境

```
conda create -n hyw_graspnet_baseline python=3.8
```

进入

```
conda activate hyw_graspnet_baseline
```

查看requirements.txt，发现需要torch1.6版本，为了置顶CUDA编译工具

用conda安装

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

验证下是否安装成功，进入python

```
>>> import torch
>>> torch.cuda.is_available()
True
```

注释掉torch

```
#torch==1.6
```

然后安装剩下的依赖

```
pip install -r requirements.txt
```

继续安装pointnet2

```
cd pointnet2
python setup.py install
```

继续安装knn

```
cd knn
python setup.py install
```

继续

```
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

ImportError: cannot import name 'NDArray' from 'numpy.typing' (/home/ubuntu/miniconda3/envs/hyw_graspnet_baseline_py38/lib/python3.8/site-packages/numpy/typing/__init__.py)

升级numpy

```
pip install --upgrade numpy
```

提示不兼容：

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
graspnetapi 1.2.11 requires numpy==1.20.3, but you have numpy 1.24.4 which is incompatible.

不管它



报错3：AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.

```
_MAX_FLOAT = np.maximum_sctype(np.float)
_FLOAT_EPS = np.finfo(np.float).eps
```

修改为：

```
_MAX_FLOAT = np.maximum_sctype(np.float64)
_FLOAT_EPS = np.finfo(np.float64).eps
```







运行

```
sh command_demo.sh
```

报错

CUDA kernel failed : no kernel image is available for execution on the device  

此时 torch不能正确读取tensor

当前GPU的算力与当前版本的Pytorch依赖的CUDA算力不匹配（3090ti算力为8.6，而当前版本的pytorch依赖的CUDA算力仅支持3.7，5.0，6.0，7.0, 7.5, 8.0）

# conda虚拟环境复现流程pyt1.12

装pytorch

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

装依赖

```
pip install -r requirements.txt
```

处理依赖报错

```


conda install -c conda-forge imageio

```





# 最终解决：conda虚拟环境复现流程pyt1.7+cudatoolkit 11.0

参考[graspnet-baseline 复现问题总结_/home/peter/graspnet-baseline/knn/src/cuda/vision.-CSDN博客](https://blog.csdn.net/SiriusasBlack/article/details/136048786)

