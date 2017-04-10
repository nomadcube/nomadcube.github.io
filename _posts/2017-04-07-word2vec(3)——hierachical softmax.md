---
layout: post
tags: 语言模型 神经网络 embedding
---


hierarchical softmax则是由Frederic Morin和Yoshua Bengio在2005年的论文*Hierarchical Probabilistic Neural Network Language Model*中提出。提出的原因显然是为了解决full softmax模型的训练性能低下问题。

基本思想是将$$P(v; w_{t-1}, ..., w_{t-n+1})$$进行分解。在分解之前，计算这个概率需要遍历词汇表。

hierarchical softmax所采用的分解方式是，将v表达成一棵平衡树的一个叶节点，这棵平衡树的每个内结点都是0或1，代表着是左子树还是右子树。这样做实质上是给每个词进行编码，编码的长度最大为m，也就是$$log(V)$$. 同时，每个内结点也可以沿着root表示成由0和1组成的一组编码。这样做的好处是，可以将$$P(v; w_{t-1}, ..., w_{t-n+1})$$按v的每一位编码进行分解了：$$P(v; w_{t-1}, ..., w_{t-n+1}) = \prod_j^m P(b_j(v);b_1(v),...,b_{j-1}(v);w_{t-1},...,w_{t-n+1})$$

分解之后，累乘中的每一项都是一个logit表达式，代表的是v的每一位编码取值为1的概率，看作是一个二分类问题，这样就不存在使用softmax时需要遍历词汇表算标准化项的问题。

但这又带来另一个问题：针对每一位的编码都需要有单独的一个logit表达式，也就是说一组独特的参数。这就导致参数空间过大或模型过拟合的倾向。

为了解决这个新的问题，他们又提出"参数共享"的概念。具体做法是，将树的内结点也看成是一个有意义的"词"，和词汇表里的词同等对待，唯一的差别在于查找这些内结点的embedding向量时对应的矩阵是另外一个。这样做的好处是将$$P(v; w_{t-1}, ..., w_{t-n+1}) = \prod_j^m P(b_j(v);b_1(v),...,b_{j-1}(v);w_{t-1},...,w_{t-n+1})$$变成已知context和内结点的向量表示时，求每个内结点为1的概率。这就相当于内结点和目标词分别对应的embedding向量的内积了。

于是，hierarchical softmax就将$$P(w_O;w_I)$$的计算从一个softmax表达式简化成$$\prod_{j=1}^m logit(u_{n(w,j)} * v_{w_I})$$, 其中u代表内结点的embedding向量，v代表叶结点的embedding向量，$$n(w,j)$$表示从root到词$$w_I$$路径上的第j个结点。也就是说，转换成了目标词向量与其路径上各个内结点向量的点积之连乘。


### 层次softmax网络结构
### 输入层到映射层
### 映射层到输出层
#### 词编码
#### 

### 参考文献
- *Hierarchical Probabilistic Neural Network Language Model*
