---
layout: post
tags: 深度学习
---

在tensorflow编程时需要定义计算图，计算图的基本组成是tensor和operation, tensor相当于数据，operation相当于运算。tensor作为operation的输入和输出。

打印tensor本身并不会输出它所包含的数据，因为定义了一个图的时候也不会即时计算出它的输出tensor所包含的值，需要调用session.run才会真正触发计算。

### placeholder
placeholder是一种tensor类型，和其它各种tensor一样，它也可以对python的各种张量形式的数据类型进行封装。而它的特点在于，它在定义时是空的，只有让python程序给它feed数据之后才能对它做evaluation. 而feed数据的形式，可以用一个dict, 即 ```{palcaholder: python值}```这样的形式。

### 读数据
主要是3种形式：用python数据去feed；从本地文件读取；直接初始化constance或variable. 可见第一种是对应于placeholder这类tensor的。
在docker中使用tensorflow时，如果要选择从本地文件读取数据，那么需要先在docker的tensorflow目录下建立文件，在当前目录启动python, 再用TextLineReader和decode_csv进行读文件及解析csv文件。

![aaa](https://github.com/nomadcube/nomadcube.github.io/blob/master/public/test_pic.png)
