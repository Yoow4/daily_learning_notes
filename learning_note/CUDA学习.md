# CUDA学习

## 核函数

1、核函数在GPU上进行**并行**执行
2、注意:
(1)限定词 global修饰

(2)返回值必须是void

3、形式

```
__global__ void kernel_function(argument arg)
{
	...
}
```

注意事项：
1、核函数只能访问GPU内存
2、核函数不能使用变长参数
3、核函数不能使用静态变量
4、核函数不能使用函数指针
5、核函数具有异步性：CPU和GPU不同架构



CUDA程序编写流程:

```
int main(void)
{
	主机代码
	核函数调用
	主机代码
	return 0;
}

```

## 线程模型

1、线程模型重要概念:
	(1)网格grid
	(2)线程块 block
2、线程分块是逻辑上的划分，物理上线程不分块
3、配置线程<<grid_size, block_size>>
4、最大允许线程块大小:1024最大允许网格大小:231-1(针对一维网格)

## nvcc编译流程

1、nvcc分离全部源代码为:(1)主机代码(2)设备代码
2、主机(Host)代码是C/C++语法，设备(device)代码是C/C++扩展语言编写
3、nvcc先将设备代码编译为PTX(Parallel Thread Execution)伪汇编代码，再将PTX代码编译为二进制的cubin目标代码
4、在将源代码编译为 PTX 代码时，需要用选项-arch=compute_XY指定一个**虚拟**架构的计算能力用以确定代码中能够使用的CUDA功能。
5、在将PTX代码编译为cubin代码时，需要用选项-code=sm_ZW指定一个**真实**架构的计算能力，用以确定可执行文件能够使用的GPU。注意真实的能力要大于虚拟的能力



### PTX

PTX(Parallel Thread Execution)是CUDA平台为基于GPU的通用计算而定义的虚拟机和指令集
nvcc编译命令总是使用两个体系结构:一个是虚拟的中间体系结构，另一个是实际的GPU体系结构
虚拟架构更像是对应用所需的GPU功能的声明
虚拟架构应该尽可能选择低----适配更多实际GPU
真实架构应该尽可能选择高----充分发挥GPU性能



## GPU计算能力

区分架构代号特斯拉(Tesla)费米(Fermi)开普勒(Kepler)麦克斯韦(Maxwell)帕斯卡(Pascal)伏特(Volta)图灵(Turing

和计算能力的版本号sm_50,sm_52 and sm_53....


并非GPU 的计算能力越高，性能就越高

性能与计算能力 显存容量、显存带宽、浮点数运算峰值有关

## CUDA程序兼容性问题

### 指定虚拟架构计算能力

1、C/C++源码编译为PTX时，可以指定虚拟架构的计算能力，用来确定代码中能够使用的CUDA功能
2、C/C++源码转化为PTX这一步骤与GPU硬件无关
3、编译指令(指定虚拟架构计算能力)

```
-arch=compute_XY
XY:第一个数字X代表计算能力的主版本号，第二个数字Y代表计算能力的次版本号
```

4、PTX的指令只能在更高的计算能力的GPU使用
例如:

```
nvcc helloworld.cu-o helloworld -arch=compute 61
```

编译出的可执行文件helloword可以在计算能力>=6.1的GPU上面执行，在计算能力小于6.1的GPLM不能执行。

### 指定真实架构计算能力

PTX指令转化为二进制cubin代码与具体的GPU架构有关
编译指令(指定真实架构计算能力)

```
-code=sm_XY
XY:第一个数字X代表计算能力的主版本号，第二个数字Y代表计算能力的次版本号
```

注意:
(1)二进制cubin代码，大版本之间不兼容!!!
(2)指定真实架构计算能力的时候必须指定虚拟架构计算能力!!!
(3)指定的真实架构能力必须大于或等于虚拟架构能力!!!

```
nvec helloworld.cu -o helloworld arch=compute_61 code-sm_60  （无法执行）
```

真实架构可以实现低小版本到高小版本的兼容!

### 指定多个GPU版本编译

使得编译出来的可执行文件可以在多GPU中执行同时指定多组计算能力:
编译选项-gencode arch=compute XY-code=sm XY

例如:

```
-gencode=arch=compute_35,code=sm_35开普勒架构  等价于-arch=sm_35
-gencode=arch=compute_50,code=sm_50麦克斯韦架构 等价于-arch=sm_50
-gencode=arch=compute_60,code=sm_60帕斯卡架构
-gencode=arch=compute_70,code=sm_70伏特架构
```

编译出的可执行文件包含4个二进制版本，生成的可执行文件称为胖二进制文件(fatbinary)


注意:(1)执行上述指令必须CUDA版本支持7.0计算能力，否则会报错
2)过多指定计算能力，会增加编译时间和可执行文件的大小

### nvcc编译默认计算能力

不同版本CUDA编译器在编译CUDA代码时，都有一个默认计算能力
CUDA 6.0及更早版本: 			默认计算能力1.0
CUDA 6.5~CUDA 8.0:			 默认计算能力2.0
CUDA 9.0~CUDA 10.2 		   默认计算能力3.0
CUDA 11.6:						  	默认计算能力5.2



