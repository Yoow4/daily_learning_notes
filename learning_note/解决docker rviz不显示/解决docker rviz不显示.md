# 解决docker中使用rviz无法显示

运行rviz时报错

```
[ERROR] [1710215116.967019965]: Unable to create the rendering window after 100 tries.
[ INFO] [1710215116.967032855]: Stereo is NOT SUPPORTED
terminate called after throwing an instance of 'std::logic_error'
  what():  basic_string::_M_construct null not valid
Aborted (core dumped)
```

运行一键安装命令：wget http://fishros.com/install -O fishros && . fishros

完成带ros的docker安装



打包该镜像， docker将容器打包成镜像的命令如下：

```
docker commit [-m="描述信息"] [-a="创建者"] 容器名称|容器ID 生成的镜像名[:标签名]
```

​        例如，咱们把名称为noetic1的容器打包成一个叫ltdz的镜像，标签名瞎起一个就行了：

```
docker commit noetic1 ltdz:15
```

​        然后查看是否打包成功，列举本地的所有镜像：

```
docker image ls
```

​        若有你刚才打包的那个，说明打包成功。

​    最后将这个镜像运行起来，命令与第三节的究极解决方案保持一致，并将镜像名改成你刚才打包成的那个镜像名：



```
sudo docker run -dit \
--name=[your_container_name] \
--privileged  \
-v /dev:/dev \
-v /home/[your_username]:/home/[your_username] \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
-e DISPLAY=unix$DISPLAY \
-w /home/[your_username] \
--net=host 
[你刚才打包出的镜像名，即ltdz:15]
```

为了在docker中使用rviz，增加了如下命令允许docker使用宿主机的显卡资源

```
--gpus all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
```

最后输入

```
sudo docker run -dit \
--name=robotic_grasp \
--privileged  \
-v /dev:/dev \
-v /home/ubuntu/data0/hyw:/home/ubuntu/data0/hyw \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
-e DISPLAY=unix$DISPLAY \
-w /home/ubuntu/data0/hyw \
--gpus all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
--net=host \
grasp1

```

启动容器运行roscore和rviz

成功

![image-20240313154143961](D:\data\docsify\hyw学习笔记\解决docker rviz不显示\解决docker rviz不显示.assets\image-20240313154143961.png)



参考连接：[针对ros机器人开发同学的docker入门教程 - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv25771461/)

[ROS docker在安装主机显卡驱动后，rviz等可视化工具报错_docker rviz::rendersystem: error creating render w-CSDN博客](https://blog.csdn.net/GritYearner/article/details/133679403)

[解决ubuntu20.04 安装docker ros melodic 无法打开rviz和gazebo 问题_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1m94y1K7ES/?spm_id_from=333.337.search-card.all.click&vd_source=bf7b9535de982f1d288138463991a3f7)