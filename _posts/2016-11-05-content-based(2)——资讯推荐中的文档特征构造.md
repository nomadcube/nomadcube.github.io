---
layout: post
tags: 推荐算法
---

最简单的做法是用TFIDF将文档转成向量空间表示。
影响效果的几个因素：

1. 切词粒度

分词粒度越大, precision越高, 即召回的文章更少, 但相对来说相关性更强。因为一些词条切分成更细粒度的多个词条之后所表达的意思和原词条并不一致, 会召回一些相关性比较低的文章。

2. IDF阈值选择
3. TFIDF是否进行标准化
4. 是否考虑连续词条拼接
