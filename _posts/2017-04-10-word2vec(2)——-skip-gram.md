---
layout: post
tags: 语言模型 神经网络 embedding
---

skip-gram的意义是将一份完整的语料观测切分成多个新的样本观测，根据这多个新的样本观测计算出word2vec的学习目标。

类似于n-gram，skip-gram也基于这么一个假设：一个词（记为target词）在语料中的出现概率主要由它周围最近的若干个词（记为context）决定，其中"skip"是指不需要将某target和context中每个词组成的对都作为样本考虑到模型中，只需要抽"number of skip"这么多个邻居。这从直觉上来看也是为了模型的平滑性。

接下来以"This inversion might seem like an arbitrary choice"这句话为例看不同参数下skip-gram得出的样本。

skip-gram主要有2个参数：

- 词窗口大小$$k$$。是指对于特定的一个target词，构造样本时只考虑$$\pm k$$范围内的词条。$$k$$越大模型越复杂，计算成本也越高。
- skip数量$$p$$。当词窗口大小设为$$k$$时，target词周围有$$2k$$个词可选择作为input, $$p$$决定的是从$$2k$$个词中选出多少个作为input. 因此很显然$$p <= 2k$$, 而且当skip数量设成$$p$$时，样本中会包含$$p$$个以target词为label的样本点。

以下是将"This inversion might seem like an arbitrary choice"切分成若干个新的样本观测的代码：

```python
import collections
import random
import collections
import random

data = "This inversion might seem like an arbitrary choice".split(" ")


def skip_gram_samples(corpus, num_skips, skip_window_size):
    assert num_skips <= 2 * skip_window_size
    inputs = list()
    labels = list()
    span = 2 * skip_window_size + 1
    words_within_windows = collections.deque(maxlen=span)
    data_index = 0
    while data_index < len(corpus):
        move_size = 1 if len(words_within_windows) > 0 else span
        for _ in range(move_size):
            words_within_windows.append(corpus[data_index])
            data_index += 1
        target = skip_window_size
        targets_to_avoid = [skip_window_size]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            inputs.append(words_within_windows[skip_window_size])
            labels.append(words_within_windows[target])
    return inputs, labels
```

其中以下代码块对应于从$$2k$$个邻居中抽出$$p$$个作为input：

```python
for j in range(num_skips):
    while target in targets_to_avoid:
        target = random.randint(0, span - 1)
    targets_to_avoid.append(target)
    inputs.append(words_within_windows[skip_window_size])
    labels.append(words_within_windows[target])
```

假如将窗口大小设为2，skip大小设成3，调用代码及输出的input和label分别如下：

```python
all_batch, all_labels = skip_gram_samples(corpus=data, num_skips=2, skip_window_size=2)
print(all_batch)
print(all_labels)
```

> ['might', 'might', 'might', 'seem', 'seem', 'seem', 'like', 'like', 'like', 'an', 'an', 'an']


> ['inversion', 'seem', 'like', 'might', 'inversion', 'like', 'seem', 'arbitrary', 'might', 'choice)', 'seem', 'like']

### 参考资料
[1] [维基百科上的skip-gram解释](https://en.wikipedia.org/wiki/N-gram#Skip-gram)

[2] *Efficient Estimation of Word Representations in Vector Space*

[3] [TF官方文档-Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)
