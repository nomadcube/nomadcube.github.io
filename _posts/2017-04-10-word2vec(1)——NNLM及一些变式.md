---
layout: post
tags: 语言模型 神经网络 embedding
---

word2vec的雏形在2003年由Yoshua Bengio等人在论文*A Neural Probabilistic Language Model*中提出，但词向量并不是文中唯一的重点，该论文更关键的是提出了神经网络语言模型（Neural Network Language Model即NNLM）. 和其它如n-gram的语言模型一样，NNLP所要表达的也是某种语言中一段序列的联合概率，但最大的特点在于，它不依赖于语料中的共现词频统计，而是利用某段子序列对下一个词进行预测的方式来实现。

同时，在这个预测的过程中，NNLM模型还用实数空间的一个向量来代表词，而不仅仅是用一个符号。这里所说的向量就是所谓的词向量，理论上来看是这个语言模型的副产物，但从实际应用上看反而词向量是更为重要的模型产出，因为词向量能反映一个词的含义、向量之间距离能反映词之间的语义关系，这对于更上层的应用而言是很重要的输入。

结合神经网络和词向量思想，很容易理解NNLM会包含有隐藏层、将词编号映射到词向量的投影层，它们可以看作是模型的核心，对输入层和输出层进行连接。假设词汇量为V，输入序列包含N个词，词向量长度为P，隐藏层神经元个数为H, 那么完整的NNLM模型前馈过程如下图所示：

![word2vec_1](/public/word2vec_1.png)

1. 投影层得到的是N个P维向量，实际上需要进一步的处理才会到隐藏层，处理方式可以是拼接成一个NP维的向量，或求得这N个向量的质心
2. 隐藏层到输出层可以看成是典型的多分类模型，因此最普通的做法是针对输出中的每个类别（即V个词汇中的任意一个词），都用softmax即$$\frac{e^{x_i}}{\sum_j e^{x_j}}$$来表示概率，因而隐藏层到输出层是一个包含H*V+V的全连接。

从这可以看出，NNLM瓶颈在于隐藏层到输出层的全连接，因为在训练的时候对于每个样本，都需要对V个词进行遍历才算得梯度。

性能限制对于模型在业界应用是一个障碍，因此在NNLM提出来之后陆续出现一些改进的想法，主要的思路都是在保证或提高精度的前提下提高训练效率。

主要是以下几个方向：

1. 去掉隐藏层，从投影层直接连续到输出层，完全当成是词向量到输出层的多类别分类来做
2. 解决计算softmax时分母中标准化项的问题，主要是hierarchical softmax的二叉树方法、negative sampling这两种解决方法

其中negative sampling相对好理解，相当于是将多类别分类转成二元分类。

hierarchical softmax就没那么直观了，可以想象成是先将原始词汇表中的V个词进行m个层级的归类，形成一个二叉树，其中根结点相当于是最大的那个类，包含了所有的词。在基于输入序列做预测时，相当于是由上而下、由粗至细的一层又一层的判断，每一层都是解决这个问题：是否属于大类$$C_i$$? 其中i指第i层。如此细分下去，直至叶节点，也就是原始的词汇。

这样做的好处是将一个类别个数为V的多类别问题转成m个大类串联的判断问题，即m个二分类问题，其中当使用平衡二叉树时，$$m=log(V)$$, 若使用hoffman编码对每个词映射到长度不一致的01编码中，复杂度可以更低。

这就是为什么hierarchical softmax能提高NNLM的训练效率。

### 参考文献

- *A Neural Probabilistic Language Model*
- *Efficient Estimation of Word Representations in Vector Space*
- *Distributed Representations of Words and Phrases and their Compositionality*
- *Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors*
