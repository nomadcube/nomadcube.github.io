---
layout: post
tags: 相似项发现
---

simhash是一种局部敏感哈希函数,与md5等哈希算法的区别在于,对于2个只在局部有差异的字符串,md5会得到2个完全不同的哈希值,而局部敏感哈希函数则会得到2个接近的哈希值。假设哈希值用二进制去表示, 那么两个哈希值接近就是指大部分位都同时是0或1,只有少数位不同。

计算文档simhash的步骤如下:

1. 确定用几位的simhash签名,假设是4位(实际应用中会比较大,否则容易发生冲突,这里用4是为了解释方便),初始化为0000
2. 分割文档,得到n个元素。用分词方式或shingle方式都可以
3. 得到文档中n个元素分别的4位的二进制hashcode
4. simhash签名的第i位由n个元素的第i位共同决定:若某个元素的第i位为1,那么simhash签名的第i位加1, 否则减1.遍历完之后,第i位若为正数,那么simhash签名的第i位为1, 否则为0, 由此得到4位simhash签名的二进制表示

用Python实现:

```python
    # 生成simhash值
    def simhash(self, tokens):
        v = [0] * self.hashbits
        for t in [self._string_hash(x) for x in tokens]:  # t为token的普通hash值
            for i in range(self.hashbits):
                bitmask = 1 << i
                if t & bitmask:
                    v[i] += 1  # 查看当前bit位是否为1,是的话将该位+1
                else:
                    v[i] -= 1  # 否则的话,该位-1
        fingerprint = 0
        for i in range(self.hashbits):
            if v[i] >= 0:
                fingerprint += 1 << i
        return fingerprint  # 整个文档的fingerprint为最终各个位>=0的和
```

其中```t&(1 << i)```对应于判断某个元素的第i位是否为1. ```<<```表示移位运算, 从```0001```到```1000```, ```&```表示按拉与运算,这里的```t```可以用十进制表示也可以用二进制表示, 无论如何, ```t&(1 << i)```的取值都是0或1, 表示以二进制表示时t的第i位是否为1.

```fingerprint += 1 << i```对应于判断minhash的第i位是否为1, 判断的依据是minhash中间数据的第i个元素为正还是为负。之所以可以用这个表达式是因为```1 << i```表示从1移位到2、4、8等, 若在十进制表示的数字上进行计算, 则相当于将对应的二进制表示对应的那一位设成1.

注: Python代码段引用自 http://blog.csdn.net/al_xin/article/details/38919361

