

```
export SPARK_SUBMIT_OPTS=-Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=7777
```

```
./spark-submit --class org.wumengling.fun.pkg.TestForFun /Users/wumengling/Documents/TestForFun/target/fun-1.0-SNAPSHOT.jar
```


