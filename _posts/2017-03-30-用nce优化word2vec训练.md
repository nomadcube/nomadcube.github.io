---
layout: post
tags: 深度学习
---

nce和full softmax最大的区别有两个：

1. 正负样本构造
2. 从隐藏层到输出层的损失函数定义

下图是full softmax从隐藏层到输出层的结构示例：
![word2vec_2](/public/word2vec_2.png)

full softmax的思想的关键点在于：

1. 完整的上下文序列作为输入，单个target词作为输出
2. 正因为1，因此隐藏层是上下文序列多个词的词向量拼接
3. 输出有V(词汇量大小)个可能的类别

导致full softmax训练性能问题的本质是多类别分类时softmax的定义：既然输出有V(词汇量大小)个可能的类别，那么在计算softmax时分母就是一个由V个单元项组成的标准化项。

nce的出现正是为了解决这个问题的，解决方法的核心是通过对负样本进行抽样：若将V个类别的多类别分类问题看成V个二元分类问题，而输入仍然是包含多个词的序列，那么在full softmax中的一条样本可以分解成1条正样本+(V-1)条负样本。而word2vec的关键在于词向量的估计，因此可以在保证词向量质量的前提下对负样本进行抽样：只取其中的K个。

如下图所示，假设上下文序列包含3个词，而且对负样本进行抽样的数目K也为3，那么word2vec的问题转化将真正的目标词t从这3个抽样所得的负样本或者说噪声中分辨出来。

![nce](/public/nce.png)

进一步地，可以将一条样本中上下文序列包含的m个词打散，成为m个单独的样本。这就将问题转换成这样的一个二元分类问题：输入为上下文序列的某个词和词汇全集中的其中一个词（可能是正样本的target或抽样出来的负样本），用lr去预测label是1还是0. 

如下图所示，原先的一条样本相当于对负样本抽样并打散后的12条样本（某些样本的label没有画出）：
![nce_1](/public/nce_1.png)

这样就可以用logit函数来表达模型，而由于词汇量仍然有V个没变，因此分类模型参数仍然是VK个权生和V个截距项，只不过在计算损失函数时只需要关注目标词t和抽样出来的K个负样本：

$$loss_{nec} = log\sigma(u_{w_o}v_{w_I}) + \sum_i^K E_{w_i from P_n(w)}log \sigma(-u_{w_i}v_{w_I})$$

其中$$\sigma$$代表logit函数，$$P_n(w)$$为负采样的分布，常取均匀分布；向量$$u$$为nec层的权重参数，$$v$$为向量嵌入层的参数。以上实际上是以$$w_I$$为输入词的K+1条样本对应的交叉熵损失函数。

### 参考资料
[1] *Distributed Representations of Words and Phrases and their Compositionality*

[2] [TensorFlow-Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec) 
