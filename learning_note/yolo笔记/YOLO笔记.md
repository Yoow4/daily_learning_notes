# YOLO笔记

## YOLOV1 

将分类预测问题变成回归问题,两阶段简化到单阶段，

### 预测过程

grid cell  、 IOU 、  NMS

每个俩个BBX

### 训练过程

### 损失函数

![image-20230211113419884](./assets/image-20230211113419884.png)

输入448 * 448 * 3

输出7 * 7 * 30

## YOLOv2    YOLO9000

### BN

### Anchor Dimension Clusters

使用5个anchor，预先设定好长宽比的框

![image-20230211113044410](./assets/image-20230211113044410.png)

### 损失函数

![image-20230211113356768](./assets/image-20230211113356768.png)

### 不同操作的效果

![image-20230211111259869](./assets/image-20230211111259869.png)

### 大图片效果更好

![image-20230211113537297](./assets/image-20230211113537297.png)

### 改进了网络结构 

darknet19

输入416 * 416 * 3

输出13 * 13 * 5 * 25

![image-20230211121041608](./assets/image-20230211121041608.png)

## YOLOv3

### 算法框架

![image-20230211122358479](./assets/image-20230211122358479.png)

![image-20230211122531069](./assets/image-20230211122531069.png)

与YOLOv2不同，此处是pc相乘之后才是confidence

![image-20230211122854955](./assets/image-20230211122854955.png)



### 损失函数

![image-20230211123634933](./assets/image-20230211123634933.png)

