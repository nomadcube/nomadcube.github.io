---
layout: post
tags: 草稿
---

tensorflow接口包括tf-core这些底层的接口，以及如tf-train这些高层次的接口。

tf-core主要是用于定义一些张量（tensor, 因此也包含标量、向量、线性算子等）以及一些node节点。

tf-train则定义了一些高层次的机器学习接口，如最优化方法等。

通常在模型定义的时候，可以用tf-core定义一些代表损失函数的图结构，之后用tf-train的最优化方法去求解。

一般在模型定义的时候，样本数据（x和y）是用placeholder来定义的，而参数如w和b则是用variable来定义的。
