# ResNet笔记

参考链接：[ResNet详解与分析 - shine-lee - 博客园 (cnblogs.com)](https://www.cnblogs.com/shine-lee/p/12363488.html)

参考视频：[ResNet论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1P3411y7nn/?spm_id_from=333.788&vd_source=bf7b9535de982f1d288138463991a3f7)



解决问题：神经网络层数变多，性能却在下降



关键：residual connection 残差连接、模型复杂度低

梯度在后续训练时较大，

原因：求导时由原来的乘法求导项F(g(x))变为对F(g(x))+g(x)加法求导，有一个较大的求导项，梯度不至于消失。