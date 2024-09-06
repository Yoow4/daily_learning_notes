# typora打开闪退ubuntu20.04解决

报错：

```
yoow@yoow-u:/opt/Typora-linux-x64$ ./Typora 
[20428:0730/105152.811421:FATAL:gpu_data_manager_impl_private.cc(439)] GPU process isn't usable. Goodbye.
追踪与中断点陷阱 (核心已转储)

```

用参数打开

```
./Typora  -no-sandbox 
```

发现可以正常工作



```
sudo gedit ~/.bashrc
```

最后一行加入

```
alias Typora='/opt/Typora-linux-x64/Typora -no-sandbox'
```

终端运行

```
source ~/.bashrc 
```



```
Typora
```

即可正常打开