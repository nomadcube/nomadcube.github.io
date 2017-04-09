---
layout: post
tags: 语言模型 神经网络
---

针对语言模型的上下文预测问题有两种建模形式，现以"问题有两种建模形式"这个句子为例：

1. CBOW, 即用上下文作为输入，目标词作为输出，如输入是"问题有两种建模"，目标是预测接下来的词是"形式"的概率
2. skip-gram, 即以目标词作为输入，上下文作为输出，如输入是"两种建模"，目标是预测它的上下文出现"问题是"、"形式"的概率

本文暂时不对这两种形式作对比，并仅以CBOW来阐述word2vec. 

### **word2vec基本形式**
在2003年由Yoshua Bengio等人在论文*A Neural Probabilistic Language Model*中提出神经网络语言模型word2vec，用网络结构去对语言进行建模，基本形式如图：

![word2vec_1](/public/word2vec_1.png)

其中embedding阶段可以认为是模型的核心部分。因为在这个模型提出之前几乎所有语言模型都用符号来表示词，词与词之间的关系用通过共现频率反映，而word2vec模型的改进点在于，用连续向量来代表词，词之间的关系用向量距离进行衡量。可见embedding给模型带来了$$VK$$个参数，其中$$V$$为词汇量大小，$$K$$为向量长度。

而在full connect阶段，则将预测问题看成是多类别分类问题，输出层的类别个数是$$V$$, 也就是需要预测词汇量大小个类别的概率，同时embed得到的词向量成为网络结构的隐藏层。这时候连接隐藏层和输出层的参数个数为$$VKV$$。

### **full softmax**
在使用BP对模型参数进行求解时会有一个问题：在模型前向传播时，输出层"似乎"只有一个词和隐藏层的连接会被用到，也就是说"似乎"隐藏层和连接层的$$VKV$$当中在一次迭代时只需要更新其中的$$KV$$个。但实际上并不是这样的。当我们使用softmax去解决这个多类别分类问题时，输出层每一个词的概率都会被表示成相对于隐藏层的$$VKV$$个参数的表达式，因为softmax是需要在分母加标准化项的：$$\frac{e^{x_i}}{\sum_j e^{x_j}}$$, 分母中的$$\sum_j e^{x_j}$$实际上是需要用到输出层每一个可能的词与隐藏层的连接的。

![word2vec_2](/public/word2vec_2.png)

以上这种求解方式也称为full softmax, 导致每次迭代都需要对$$VKV+VK$$个参数进行求梯度和参数更新，会影响模型训练性能。为了解决这个问题，目前主要使用的两种思路分别是hierarchical softmax和NCE。

### **hierarchical softmax**
hierarchical softmax则是由Frederic Morin和Yoshua Bengio在2005年的论文*Hierarchical Probabilistic Neural Network Language Model*中提出。提出的原因显然是为了解决full softmax模型的训练性能低下问题。

基本思想是将$$P(v; w_{t-1}, ..., w_{t-n+1})$$进行分解。在分解之前，计算这个概率需要遍历词汇表。

hierarchical softmax所采用的分解方式是，将v表达成一棵平衡树的一个叶节点，这棵平衡树的每个内结点都是0或1，代表着是左子树还是右子树。这样做实质上是给每个词进行编码，编码的长度最大为m，也就是$$log(V)$$. 同时，每个内结点也可以沿着root表示成由0和1组成的一组编码。这样做的好处是，可以将$$P(v; w_{t-1}, ..., w_{t-n+1})$$按v的每一位编码进行分解了：$$P(v; w_{t-1}, ..., w_{t-n+1}) = \prod_j^m P(b_j(v);b_1(v),...,b_{j-1}(v);w_{t-1},...,w_{t-n+1})$$

分解之后，累乘中的每一项都是一个logit表达式，代表的是v的每一位编码取值为1的概率，看作是一个二分类问题，这样就不存在使用softmax时需要遍历词汇表算标准化项的问题。

但这又带来另一个问题：针对每一位的编码都需要有单独的一个logit表达式，也就是说一组独特的参数。这就导致参数空间过大或模型过拟合的倾向。

为了解决这个新的问题，他们又提出"参数共享"的概念。具体做法是，将树的内结点也看成是一个有意义的"词"，和词汇表里的词同等对待，唯一的差别在于查找这些内结点的embedding向量时对应的矩阵是另外一个。这样做的好处是将$$P(v; w_{t-1}, ..., w_{t-n+1}) = \prod_j^m P(b_j(v);b_1(v),...,b_{j-1}(v);w_{t-1},...,w_{t-n+1})$$变成已知context和内结点的向量表示时，求每个内结点为1的概率。这就相当于内结点和目标词分别对应的embedding向量的内积了。

于是，hierarchical softmax就将$$P(w_O;w_I)$$的计算从一个softmax表达式简化成$$\prod_{j=1}^m logit(u_{n(w,j)} * v_{w_I})$$, 其中u代表内结点的embedding向量，v代表叶结点的embedding向量，$$n(w,j)$$表示从root到词$$w_I$$路径上的第j个结点。也就是说，转换成了目标词向量与其路径上各个内结点向量的点积之连乘。

### **NCE**
NCE则利用采样的思想。目标词和context的正样本是直接从语料中构造出来的，但负样本量很大，以window大小设成1为例，这样负样本相当于是目标词和词汇表的其它的n-1个词的组合。NCE的做法是从中进行抽样，只以其中的k个作为真实使用的负样本。预测目标是使得正样本的对数似然值高，而负样本的比较低，概率是依据logistic算出来的。

### 参考文献

- *A Neural Probabilistic Language Model*
- *Distributed Representations of Words and Phrases and their Compositionality*
- *Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors*
- *Hierarchical Probabilistic Neural Network Language Model*
- *Learning word embeddings efficiently with noise-contrastive estimation*
