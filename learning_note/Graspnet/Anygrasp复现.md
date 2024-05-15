# Anygrasp复现



```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch --no-pin

```



安装openblas-devel时，它自动又安装了cpu版本的pytorch，从而导致pytorch无法调用GPU，torch.cuda.is_available()为False，我删掉环境重新安装pytorch和这个依赖都是如此，无论是先安装openblas-devel依赖还是先安装GPU版本的pytorch，都出现pytorch安装成cpu版本的问题。

解决办法：清除conda中下载的包的缓存，让这个依赖真正重新下载而不是从cache中安装

```
conda clean --all --yes


```

发现没用，



## 安装MinkowskiEngine

### 先装openblas再装pytorch

```
conda install openblas-devel -c anaconda # 安装依赖

```

发现还是装了cpu的torch

1.12.1的没法覆盖安装，

尝试1.12.0的torch发现安装成功

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

进入python利用命令

```
>>> import torch
>>> torch.cuda.is_available()
```

发现没装成功

又回头装了

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

将仓库clone下来本地安装

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

编译结果

```
Processing MinkowskiEngine-0.5.4-py3.9-linux-x86_64.egg
creating /home/ubuntu/miniconda3/envs/hyw_anygrasp/lib/python3.9/site-packages/MinkowskiEngine-0.5.4-py3.9-linux-x86_64.egg
Extracting MinkowskiEngine-0.5.4-py3.9-linux-x86_64.egg to /home/ubuntu/miniconda3/envs/hyw_anygrasp/lib/python3.9/site-packages
Adding MinkowskiEngine 0.5.4 to easy-install.pth file

Installed /home/ubuntu/miniconda3/envs/hyw_anygrasp/lib/python3.9/site-packages/MinkowskiEngine-0.5.4-py3.9-linux-x86_64.egg
Processing dependencies for MinkowskiEngine==0.5.4
Searching for numpy==1.22.3
Best match: numpy 1.22.3
Adding numpy 1.22.3 to easy-install.pth file
detected new path './MinkowskiEngine-0.5.4-py3.9-linux-x86_64.egg'
Installing f2py script to /home/ubuntu/miniconda3/envs/hyw_anygrasp/bin
Installing f2py3 script to /home/ubuntu/miniconda3/envs/hyw_anygrasp/bin
Installing f2py3.9 script to /home/ubuntu/miniconda3/envs/hyw_anygrasp/bin

Using /home/ubuntu/miniconda3/envs/hyw_anygrasp/lib/python3.9/site-packages
Searching for torch==1.12.1.post200
Best match: torch 1.12.1.post200
Adding torch 1.12.1.post200 to easy-install.pth file
Installing convert-caffe2-to-onnx script to /home/ubuntu/miniconda3/envs/hyw_anygrasp/bin
Installing convert-onnx-to-caffe2 script to /home/ubuntu/miniconda3/envs/hyw_anygrasp/bin
Installing torchrun script to /home/ubuntu/miniconda3/envs/hyw_anygrasp/bin

Using /home/ubuntu/miniconda3/envs/hyw_anygrasp/lib/python3.9/site-packages
Searching for typing-extensions==4.11.0
Best match: typing-extensions 4.11.0
Adding typing-extensions 4.11.0 to easy-install.pth file

Using /home/ubuntu/miniconda3/envs/hyw_anygrasp/lib/python3.9/site-packages
Finished processing dependencies for MinkowskiEngine==0.5.4
```

测试一下

```
>>> import MinkowskiEngine as ME
/home/ubuntu/data0/hyw/anygrasp_sdk/MinkowskiEngine/MinkowskiEngine/__init__.py:36: UserWarning: The environment variable `OMP_NUM_THREADS` not set. MinkowskiEngine will automatically set `OMP_NUM_THREADS=16`. If you want to set `OMP_NUM_THREADS` manually, please export it on the command line before running a python script. e.g. `export OMP_NUM_THREADS=12; python your_program.py`. It is recommended to set it below 24.
  warnings.warn(
```

查询发现这是一个用户警告，提示你在运行 MinkowskiEngine 时，`OMP_NUM_THREADS` 环境变量没有设置。果断忽略，继续，成功输出

```
>>> print(ME.__version__)
0.5.4
```

## 装依赖库

```
 pip install -r requirements.txt
```

报错由于 `graspnetAPI` 中引用了 `sklearn` 而不是 `scikit-learn` 导致的。不知道怎么改。下载graspnetAPI仓库，在本地安装。

```
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

