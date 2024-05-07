# ViT

Vision Transformer

视觉任务当成语言处理任务去做



Transformer 与卷积神经网络相比缺少两个归纳偏置（inductive biases），即少了先验知识:

①locality，卷积是用滑动窗口的，它假设图片上相邻的区域会有相邻的特征

②translation equivariance,平移同变性，即f(g(x))=g(f(x)),意味着卷积和平移两个操作哪个先都不影响