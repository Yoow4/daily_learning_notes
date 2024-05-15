# setup.py如何构建

setuptools



1、找到源代码中个的c/c++代码，梳理出C源码和整个项目的module的对应关系；

2、找到系统当中的编译器，编译上述代码的.so

3、.so文件

正确编译的.so文件相当于python中的与个module



包名name

文件路径source

依赖depends

依赖的头文件include_dir









```
name
version
description
packages

需要处理的单文件模块列表
py_moudules=[ ]
作者/作者邮箱
author/author_email
长描述
long_description=""
依赖的其他包
install_requires=[ ]
项目主页地址
url

```

