# Docker快速入门

## 基本概念

打包：环境变成安装包

分发：上传到镜像仓库

部署：一个命令自动模拟环境运行应用



## 基本指令

### **指令大全**：

[nvidia-docker配置深度学习环境服务器（cuda+cudnn+anaconda+python）_冰雪棋书的博客-CSDN博客_nvidia-docker](https://blog.csdn.net/zml194849/article/details/110822831)

### 镜像部分

```
搜索镜像：docker search 
获取镜像： docker pull
查看镜像：docker images
删除镜像：docker rmi
```



### 启动&停止&重启

```
docker start id
docker stop id
docker kill id #强制停止
systemctl start docker
```

### 查看容器启动情况

```
docker ps
```

### 进入容器

```
docker exec -it <名字或ID> /bin/bash
```



### 复制文件

[Docker和Ubuntu主机互传复制文件_51CTO博客_docker复制文件到宿主机](https://blog.51cto.com/shijianfeng/5117131)

```
#宿主机到容器
docker cp 本地文件路径 ID全称:容器路径
#容器到宿主机
docker cp ID全称:容器文件路径 本地路径

```

