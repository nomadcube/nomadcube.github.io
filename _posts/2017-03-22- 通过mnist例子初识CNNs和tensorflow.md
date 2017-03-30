---
layout: post
tags: 深度学习
---

mnist是典型的图像分类任务，类别对应于0~9的这10个数字，输入则是若干张灰度图。至于为什么用灰度图就可以呢？直观来看，是因为我们只需要根据轮廓就可以区分出图中是哪个数字，即使用上了彩色图，其中的色彩数据也不会给这个任务带来更多有用的信息。

### **softmax lr及在tensorflow上的实现**
因此mnist可以看成是以灰度图为输入，0~9其中一个数字为输出的多类别分类任务。对于多类别的分类任务来说，最简单的模型莫过于softmax lr, 也就是说以图像中每个像素点的灰度值作为特征，参数大小为$$n*n*1*10$$, 其中$$n*n$$为图片的像素大小。在学习时只需要以交叉熵作为最优化目标函数，也就是说求$$\sum_i -y_ilog \hat{y_i}$$的最小值解，其中$$\hat{y_i} = logit(wx+b)$$. 

![softmax-lr](/public/softmax-lr.png)

以下为softmax lr在tensorflow的实现代码（主要借鉴自tensorflow官方文档）

{% highlight python %}
# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# x和y的place holder
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 初始化参数权重和阈值权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 训练和测试
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
{% endhighlight %}

结合tf特性对这段程序作一些解读：

1. 由于tensorflow的声明式编程和延迟计算特性，从定义session那一行之前都属于计算图的声明，之后才是真正执行计算图的计算过程
2. 在执行计算过程时才喂训练数据给程序，也就是在开头定义的两个placeholder类型的x和y
3. 计算图和机器学习算法的要素一一对应，其中tensor包括训练数据和模型参数x, y, w, b. 它们构成计算图的数据输入；operations包括inference、计算loss、最优化求解，分别对应于统计机器学习算法的三要素：模型、策略、求解算法

### **softmax lr的问题**
在mnist数据上，softmax lr模型一般只能得到90%左右的精度，这是比较惨的表现。softmax lr的问题在于：

1. 不能反映各个像素的位置关系。因为lr模型将每个图像的特征都拉平成一个$$n*n$$维向量了, 两个像素之间是否相邻并不会影响模型决策
2. 是一个关于各个像素灰度值的线性模型，没有考虑到像素之间的非线性关系

因此在典型的图像分类任务中最常使用的模型是卷积神经网络(CNNs)而不是softmax lr, 因为CNNs正是对处于图像中"某一块"的局部图像进行映射或抽样，将处理完的张量再拉平成一个向量喂给一个fully connect层。这个fully connect层实质上就是一个softmax lr. 因此CNNs的关键在于fully connect层之前的映射或抽样，也就是说对局部非线性特征的提取和处理。

### **CNNs**
一个只有单个卷积层的CNN结构如图：

![cnn](/public/cnn.png)

这个结构的定义对应于代码中"定义卷积和max pooling"、"定义full connect层"、"定义输出层"三部分：

{% highlight python %}
# coding=utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 训练数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义卷积和max pooling
pitch_size = 5
num_filters = 32
stride_size = 1
conv_w = weight_variable([pitch_size, pitch_size, 1, num_filters])
conv_b = bias_variable([num_filters])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, conv_w, strides=[stride_size, stride_size, stride_size, stride_size],
                                  padding='SAME') + conv_b)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义full connect层
W_fc1 = weight_variable([14 * 14 * 32, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 定义输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# 训练
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

{% endhighlight %}

从流程图中也可看出CNN和softmax lr最大的区别在于多了卷积计算、ReLU激活、pooling. 

卷积计算所做的相当于将图像映射成一个"更深"的张量，如以上代码是将```[-1, 28, 28, 1]```的张量映射成了```[-1, 28, 28, 32]```. 而这里的32指的是卷积层用了32个filter, 每一个filter都去捕捉原始图像的局部信息。

ReLU激活是简单地将小于0的值都不激活，这就使得这个```[-1, 28, 28, 32]```只有部分"小格子"是继续发挥作用的。

pooling要做的与张量的深度无关，只与它的长和宽有关。2*2 max pooling是将原始图像中任何一个 2*2 格子都只挑值最大的那个进入下一层。从直观上理解相当于只挑那些最显著的图像特征。因此数据量也随之缩小到```[-1, 14, 14, 32]```.

这也就是为什么说深度学习是在替代人工做特征工程了，比如说将一个单通道的灰度图像映射到32个filter的张量，这个过程就相当于对原始图像的同一个位置进行32次的抽象；多个隐藏层叠加就相当于是对原始图像进行逐层抽象，一步一步深入。

以下是在laptop上训练一个CNN模型的过程，指的是每一次迭代在当前mini-batch个图片上的准确率。这里使用的mini-batch大小是50张图片，共迭代10000次，也就是说这会遍达500000次图片，如果总共有20000张图，那么这个网络的epoch量就是25. 一般来说在CNN中，mini-batch越大，会使得模型变差。

![cnn_training](/public/cnn_training.png)

参考资料
[1] http://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network

[2] https://www.tensorflow.org/get_started/mnist/pros

