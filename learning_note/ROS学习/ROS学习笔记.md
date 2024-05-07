# ROS学习笔记



## 关键组件

### Launch文件

![image-20230224203320365](./assets/image-20230224203320365.png)

ns(namespace命名空间)

![image-20230224203637940](./assets/image-20230224203637940.png)

param是ROS整个参数服务器中使用，相当于全局变量

arg则仅在launch中使用

![image-20230224203757709](./assets/image-20230224203757709.png)

remap相当于把别人做好的功能包重命名，改别人的接口，但要小心，所有from的名称都会变成to后的名称，全局改变



### TF坐标变换

![image-20230224204159866](./assets/image-20230224204159866.png)

能保存10s的时间跨度

数据结构以树形结构保存

**tf功能包安装及例程：**

![image-20230224205337127](./assets/image-20230224205337127.png)

![image-20230224204953033](./assets/image-20230224204953033.png)

![image-20230224205044244](./assets/image-20230224205044244.png)

#### 编译代码部分

![image-20230224205249439](./assets/image-20230224205249439.png)



### QT工具箱

```
rqt_console 日志输出工具
rqt_graph 计算图可视化工具
rqt_plot 数据绘图工具
rqt_reconfigure 参数动态配置工具 
	运行方式: rosrun rqt_reconfigure
```

可以在终端输入rqt_然后TAB尝试其他工具

### Rviz可视化平台

可视化界面

Rviz插件机制

### Gazebo物理仿真环境

 ![image-20230225162458462](./assets/image-20230225162458462.png)

 ![image-20230225162622560](./assets/image-20230225162622560.png)



## 小结

![image-20230225162804315](./assets/image-20230225162804315.png)