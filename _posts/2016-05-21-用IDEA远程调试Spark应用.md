---
layout: post
tags: Spark 杂项
---

针对的场景是，用本地的idea对在远程环境中运行的Spark应用进行调试。

实现原理很简单。Spark应用在远程服务器上提交并运行后，驱动器上会启动一个JVM，所以要实现远程调试，只需要两个条件：

- 远程JVM提供被监听的端口
- idea能通过远程JVM的IP和端口进行连接

另一方面，由于Spark应用是在打包并提交后再开始调试的，因此除了以上2个条件之外，还有一个额外的前提：对Spark应用的源码在编译前先设置好断点。
因此总的来说，要实现远程调试需要以下的步骤：

1. 在待调试的源码中设置好断点

2. 在远程环境中设置Spark提交参数。即修改变量SPARK_SUBMIT_OPTS```export SPARK_SUBMIT_OPTS=-Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=7777```。其中address相当于这个Spark应用运行时对应的执行器JVM端口，suspend=y表示应用提交后先不执行，等待ide连接成功后再执行。

3. 在本地idea设置远程调试参数：先点击"Edit Configuration"后，在对话框中点击"+"号，用来新增一个remote运行配置。对于要新增的remote配置，只需要将Port参数改为在2中设置的address值，其它参数不用改。

4. 在执行2的shell下执行```./spark-submit --class org.wumengling.fun.pkg.TestForFun /Users/wumengling/Documents/TestForFun/target/fun-1.0-SNAPSHOT.jar```

这4步完成，提交之后正常情况下会看到```Listening for transport dt_socket at address: 7777```，表示这个JVM进程已经被监听，其中端口为7777。这时候就可以转到本地idea开始debug了。
