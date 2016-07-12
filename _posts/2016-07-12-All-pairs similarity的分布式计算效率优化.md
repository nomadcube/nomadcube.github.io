---
layout: post
tags: 推荐算法
---

在推荐系统中经常要计算实体之间的相似度。

记实体数目为$$M$$, 特征维数为$$N$$, 实体对应的非零特征数目最大为$$L$$，实体特征矩阵为$$A_{M,N}$。求实体两两之间的相似度最基础的步骤是要先求$$A^TA$$, 然后再根据不同的相似度度量做后续的处理。

在分布式环境下，假设有足够的机器能支持较大的并行度，那么计算$$A^TA$$的瓶颈主要在于shuffle size. 比如最简单粗暴的方法，将矩阵$$A$$按行分块存储于不同的节点，计算all-pairs相似度时需要获取存储在不同节点上的两行，这就带来$$O(n^2)$$数量级的shuffle，并且没有利用上相似度矩阵的对称性，相当于多了一倍的计算量。

当driver所在的节点能存储下矩阵$$A_{M,N}时，可以将矩阵$$A_{M,N}作为广播变量发送到各个节点，这样就能避免跨节点算相似度时带来的shuffle. 但问题在于一般真正用于生产的推荐系统的物品特征矩阵都不小，这也是使用分布式计算的初衷。

### 按行并行/按列并行

### sampling方法（DISCO/DIMSUM v1/DIMSUM v2）

### 在不同场景下各算法的复杂度分析（时间、空间、shuffle size、reduce-key complexity）

参考资料

- [Dimension Independent Similarity Computation](https://arxiv.org/pdf/1206.2082v4.pdf)
- [Dimension Independent Matrix Square using MapReduce](https://arxiv.org/pdf/1304.1467v4.pdf)
