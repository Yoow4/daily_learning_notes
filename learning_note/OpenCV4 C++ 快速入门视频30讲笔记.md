# OpenCV4 C++ 快速入门视频30讲笔记

## 001图像读取与显示

```c++
int main(int argc, char** argv) {
	Mat src = imread("D:/images/example.png",IMREAD_UNCHANGED); //  B, G, R，可选项
	if (src.empty()) {
		printf("could not load image....\n");
		return -1;
	}
	namedWindow("输入窗口", WINDOW_FREERATIO);
	imshow("输入窗口", src);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
```

主要学习:**imread(),imshow(),namewindow()**



imread()                图像读取 
             IMREAD_GRAYSCALE                 加载灰度图像
  			IMREAD_UNCHANGED                加载图像通道不变，可以加载透明通道  

imshow()               图像显示
src.empty()           判读src是否为空
namedWindow()   创建窗口
watiKey()                图像显示时间
destroyAllWindows()        销毁所有窗口



## 002色彩空间与转换

hsv: h 0~180 s 0~255 hs调整颜色, v 0~255 调整亮度

BGR:蓝绿红，注意通道顺序

```c++
cvtColor()                      色彩空间转换函数
            COLOR_BGR2GRAY              彩色到灰度

            COLOR_GRAY2BGR               灰度到彩色

            COLOR_GRAY2HSV              彩色到HSV

            COLOR_HSV2GRAY              HSV到彩色
imwirte()                        图像保存，第一个参数是图像保存路径，第二个参数是图像内存对象
```



## 003图像对象创建和赋值



创建方法：克隆、复制、赋值、创建空白图像Mat::zeros、或

```c++
// 创建方法 - 克隆
Mat m1 = src.clone():
// 复制
Mat m2;
src.copyTo(m2):
// 赋值法
Mat m3=src;

//创建空白图像
Mat m4 = Mat.zeros(src.size(), src.type()):
Mat m5 = IVt::zeros(Size(512, 512),CV_8UC3)://8代表8x8矩阵，UC代表unchar,3表示三通道
Mat m6 =Mat::ones(Size(512, 512), CV 8UC3)://ones时只有第一个通道时1

Mat kernel = (Mat_<char>(3, 3) < < 0, -1 0.
-1, 5,-1,
0, -1, 0);
```

慎用赋值

```c++
m1.clone()                克隆

m1.copyTo()                复制

Scalar()                各通道赋值

m1.channels()             通道数
/*
		copyTo 是深拷贝，但是否申请新的内存空间，取决于dst矩阵头中的大小信息是否与src一至，若一致则只深拷贝并不申请新的空间，否则先申请空间后再进行拷贝。
		clone 是完全的深拷贝，在内存中申请新的空间
	*/

```

