---
layout: post
tags: 语言模型 神经网络 embedding 开源框架
---

在[NNLM及一些变式](https://nomadcube.github.io/2017/04/10/word2vec(1)-NNLM%E5%8F%8A%E4%B8%80%E4%BA%9B%E5%8F%98%E5%BC%8F/)中提到negative sampling是其中一种优化NNLM训练性能的思路，其中最常用的negative sampling是Noise-Contrastive Estimation(NCE)，主要的作用是避免softmax时训练性能受限于词汇量大小。

使用NCE的word2vec模型结构如下图所示：

![word2vec_nce](/public/word2vec_nce.png)

可见，与[NNLM及一些变式](https://nomadcube.github.io/2017/04/10/word2vec(1)-NNLM%E5%8F%8A%E4%B8%80%E4%BA%9B%E5%8F%98%E5%BC%8F/)中的NNLM结构最大的区别在于：

1. 去除了隐藏层，从投影层直接连接到输出层
2. 输出层不再包含词汇量V个词，只包含1+k个词，其中包含了1个正例和k个抽样而得的负例

其中第2点是NCE的精髓。以skip-gram模型为例，输入为target词，输出为context. 因为这样由单个(context, target)对所对应的最优化目标函数就不再基于全量词汇，只基于这1+k个词：

$$LogLikelihood(w_O, w_{NEC}^{(1)}, ..., w_{NEC}^{(k)}, w_I; \Theta, U) = log \sigma(\Theta_{w_O} U_{w_I}) + \sum_i^k log \sigma(-\Theta_{w_{NEC}^{(i)}}U_{w_I})$$

以上为单个target词对应的样本对应的对数似然函数，解释一下：

1. 其中$$w_{NEC}^{(1)}, ..., w_{NEC}^{(k)}$$为k个负例，在实际应用时一般从context中进行抽样得到
2. $$\Theta$$和$$V$$分别对应NCE的权重参数、词向量组成的矩阵，共同组成模型的参数
3. $$\sigma$$代表logit函数
4. 对于k个负样本，在对数似然函数中可以写成求和项，是基于对context的独立同分布假设，即认为$$w_{NEC}^{(1)}, ..., w_{NEC}^{(k)}$$中的每个负例来自词袋模型的

这样NCE的权重能数$$\Theta$$仍然是$$PV$$维的，其中$$P$$为词向量长度，$$V$$为词汇量大小。假如将NNLM模型去掉隐藏层，那么投影层到输出层的连接权重也是$$PV$$维，优化点是在于将softmax改成了logit, 避免在算softmax分母的标准化项时需要用到V个词对应的权重向量。

在用tensorflow实现nce + skip-gram 结构的word2vec模型时，首先按[word2vec(2)——skip gram](https://nomadcube.github.io/2017/04/10/word2vec(2)-skip-gram/)的方式构造出样本, 接下来就可以基于这些样本构造tensorflow中的计算图（代码为tensorflow的示例代码）：

```python
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))
```

其中```num_sampled=num_sampled```代表的是NCE抽样个数，有别于skip-gram模型本身的参数num_skip. 

### 参考资料
[1] *Learning word embeddings efficiently with noise-contrastive estimation*

[2] *Distributed Representations of Words and Phrases and their Compositionality*

[3] [TensorFlow-Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec) 

