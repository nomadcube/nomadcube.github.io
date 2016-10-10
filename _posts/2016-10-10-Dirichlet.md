---
layout: post
tags: 机器学习
---

在《beta先验分布》中提到了传说中的beta分布, 当beta分布中的$$x$$从标量扩展到多维向量时, beta分布也可扩展为Dirichlet分布。换言之, Dirichlet分布是beta分布在多维上的扩展。因此两者性质也比较接近, 比如:

1. 当beta分布的2个参数相等, 即$$\alpha = \beta$$时, beta分布是对称分布。对Dirichlet分布来说也一样, 当参数$$\alpha_1, ..., \alpha_k$$都相等时, 这样的Dirichlet分布也称为对称Dirichlet分布, 从几何上来看是呈对称形态的。

2. 当beta分布的2个参数同时等于1时, beta分布退化为均匀分布。当Dirichlet分布的参数$$\alpha_1, ..., \alpha_k$$都等于1时, 同样也退化为高维空间上的均匀分布。

beta分布的物理意义是表示一组均匀分布的观测数据中某个顺序统计量的分布, 因而Dirichlet分布也类似地看成是同样条件下多个顺序统计量的联合概率密度分布。

比较常用的是对称Dirichlet分布, 即参数向量的分量都相等。这时候各分量的值也称为concentrate值。当concentrate值大于1时, 对称Dirichlet分布相对"密集", 即来自这个分布的一条观测向量之间的分量集中在某个值的可能性比较大, 当concentrate值小于1时则相反,观测向量中很可能某个分量出现的概率接近1, 其它的接近0. 但需要留意的一点是, 目前在SPARK1.6.2版本实现的LDA中, 将concentrate值限定为大于1, 具体原因尚待探讨。


