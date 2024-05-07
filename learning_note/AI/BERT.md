# BERT:

参考视频：[BERT 论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1PL411M7eQ/?spm_id_from=333.788&vd_source=bf7b9535de982f1d288138463991a3f7)

Bidirectional Encoder Representation from Transformers

关注双向性带来的好处及影响





无标号的数据上进行预训练



## 两种任务

Masked Language Model（MLM，遮盖语言模型）

Next Sentence Prediction（NSP，下一句预测）

## 三种嵌入

token嵌入（token embedding）、位置嵌入（position embedding）和段落嵌入（segment embedding）。这三个嵌入是分别计算得到的，然后在输入层直接相加起来。

这样做的原因是因为BERT模型需要同时考虑输入的token、位置和段落信息。这三个嵌入分别对应了输入token在句子中的位置和所处的段落，以及输入token本身的信息。因此，在将它们相加之后，BERT模型可以同时获得这些信息。

## 优缺点：

强大的语言表示能力：BERT模型使用了双向Transformer结构，能够学习到更丰富的语言表示，可以应对各种自然语言处理任务。

预训练模型的通用性：BERT模型是一种预训练模型，能够在大规模无标注数据上进行预训练，然后在有标注数据上进行微调，可以适用于各种任务和语言。
BERT模型也存在一些缺点：

模型过于复杂：BERT模型拥有数亿个参数，需要在GPU或TPU等硬件平台上进行训练和推理，对计算资源的要求较高。

学习时间较长：由于BERT模型需要进行预训练，因此其训练时间相对较长，需要耗费大量的计算资源。

预训练数据集需要大规模无标注数据：BERT模型的预训练需要大规模无标注数据集，这对于资源有限的组织和个人来说可能是一个限制因素



## 如何运用在生成模型中

①Seq2Seq：

​		使用BERT模型作为编码器，将输入序列转换为上下文向量，然后将上下文向量输入到解码器中进行解码生成输出序列。

②GPT模型：预训练使用BERT编码，Transformer解码

​		使用BERT模型进行预训练，然后将预训练好的BERT模型作为初始参数，使用Transformer解码器进行微调，从而实现对文本生成任务的优化。



？不太懂