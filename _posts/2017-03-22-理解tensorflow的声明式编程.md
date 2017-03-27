---
layout: post
tags: 草稿
---

作为平常主要基于Spark做机器学习应用开发的人，在刚开始接触tensorflow时挺不习惯的。比如说，为什么一定要定义一个session、为什么要在session里面"喂"数据、为什么训练数据要放在一个placeholder里面。

也许这些疑问在有经验的tf使用者看来都很低级，但对初学者来说确实是一个坎，而且是一个必须跨过的坎，不然在使用时只能停留在不知其所以然的阶段。

理解这点的关键在于意识到tf使用的是声明式编程模式的，这就意味着在编程时首先基于张量和操作定义一个计算图，这时候张量可以是赋值的也可以是未赋值的，当计算图定义好之后再整体进行计算。这时候所做的就是将计算图放到一个session里面，同时给张量赋值，并根据计算图所定义的操作进行计算。

这种逻辑和机器学习的逻辑是一致的。具体来说就是，机器学习本质上是利用训练数据去更新参数值，更新的依据是使得目标函数最优。这里所提到的"训练数据"和"参数值"就对应于tf里面的张量概念，而"目标函数最优"就对应于一个完整的计算图。也正是因为这样，我们在用tf求解一个机器学习算法的参数时，训练数据可以定义成一个placeholder类型的节点，这样就可以在计算图执行时才给训练数据所在的张量赋值；而模型参数可以定义为variable，因为在迭代的过程中需要对参数值进行更新。

在定义计算图时所依据的是具体的模型形式和求解方式，大体可以看成是inference(预测公式)、loss(损失函数定义)、optimization(最优化策略)三个部分，它们各自都相当于是计算图中的一个子树，依次使得树越来越接近完整。最美的一点在于，每一次声明都像是在写公式，但实际上对应的是树节点的定义，也就是说每一步都是在构建计算图。

### placeholder定义 
placeholder是一种tensor类型，和其它各种tensor一样，它也可以对python的各种张量形式的数据类型进行封装。而它的特点在于，它在定义时是空的，只有让python程序给它feed数据之后才能对它做evaluation. 而feed数据的形式，可以用一个dict, 即 ```{palcaholder: python值}```这样的形式。

### 读数据
主要是3种形式：用python数据去feed；从本地文件读取；直接初始化constance或variable. 可见第一种是对应于placeholder这类tensor的。
在docker中使用tensorflow时，如果要选择从本地文件读取数据，那么需要先在docker的tensorflow目录下建立文件，在当前目录启动python, 再用TextLineReader和decode_csv进行读文件及解析csv文件。

### 参考文献
*MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems*
