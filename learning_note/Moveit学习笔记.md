# Moveit学习笔记

安装Moveit

```
sudo apt-get update
sudo apt-get install ros-melodic-moveit
```



删除原有Cmakelists.txt后编译catkin_UR5包报错

```
Could not find a package configuration file provided by "soem" with any of
soemconfig.cmake
soem-config.cmake
```

解决方法：

```
sudo apt-get install ros-melodic-soem
```

其余类似错误，按照报错名字修改上述命令ros-melodic-后的部分即可



编译完成后，

```
source devel/setup.bash
```



加载Configuration出错

：QstandardPaths: XDG RUNTIME DIR not set, defaulting to '/tmp/runtime-root‘

解决：通过设置`XDG_RUNTIME_DIR`环境变量来手动指定XDG运行时目录。

```
export XDG_RUNTIME_DIR=/desired/directory/path

export XDG_RUNTIME_DIR=/desired/directory/path

```

