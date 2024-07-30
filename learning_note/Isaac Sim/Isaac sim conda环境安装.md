# Isaac sim conda环境安装

安装过程参考链接：https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html

## 安装过错报错：

运行

```
pip install isaacsim==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```

报错：

```
nvidia_stub.error.InstallFailedError:
  *******************************************************************************
  
  The installation of isaacsim for version 4.1.0.0 failed.
  
  This is a special placeholder package which downloads a real wheel package
  from https://pypi.nvidia.com. If https://pypi.nvidia.com is not reachable, we
  cannot download the real wheel file to install.
  
  You might try installing this package via
```
  $ pip install --extra-index-url https://pypi.nvidia.com isaacsim
  ```
  
  Here is some debug information about your platform to include in any bug
  report:
  
  Python Version: CPython 3.10.14
  Operating System: Linux 5.15.0-107-generic
  CPU Architecture: x86_64
  Driver Version: 555.42
  CUDA Version: 12.5
  
  *******************************************************************************
  
  [end of output]
note: This error originates from a subprocess, and is likely not a problem with pip. error: metadata-generation-failed

× Encountered error while generating package metadata. ╰─> See above for output.

note: This is an issue with the package mentioned above, not pip. hint: See above for details. 
  ```

参考链接：

https://forums.developer.nvidia.com/t/isaac-sim-python-environment-installation-with-pip-through-conda/294913

发现是pip 需要 GLIBC 2.34+ 版本兼容性。

但是

```
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC_
```

发现ubuntu20.04最高只支持2.30

解决方案：

参考链接：https://blog.csdn.net/shelutai/article/details/132363838

添加一个高级版本系统的源，直接升级libc6.

编辑源

```
sudo gedit /etc/apt/sources.list
```

添加高版本的源
deb http://mirrors.aliyun.com/ubuntu/ jammy main #添加该行到文件
运行升级

```
sudo apt update
sudo apt install libc6
```

查看结果

```
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC_
```



随后即可正常安装

## 查看安装路径

安装路径可以通过命令` pip show isaacsim `查询。

```
  pip show isaacsim
```

