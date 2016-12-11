# 20161211

这周已经在预习期末考试了，在ML上花的时间不多

## 理论部分

### 第9周 Anomaly Detection

这部分介绍了使用高斯分布和多元高斯分布做Anomaly Detection的方法

- 当样本数量不多，正样本和负样本数量不成比例的情况下（负样本出现次数远小于正样本），可以用Anomaly Detection预测负样本
- 训练只使用正样本
- 使用单纯的高斯分布，速度快，但是需要手工处理一些相关特征效果更好
- 多元高斯分布可能会出现无解的情况，此时要手动去除线性相关的特征量

## 实践部分

### 第7周SVM的编程作业

在ex6目录下

- 线性SVM分类器
- 高斯核函数SVM分类器
- 应用:垃圾邮件分类

## 坑（还是上周留的慢慢填）

- 熟悉一下numpy/matplotlib
- 用Python实现一下SVD的图像压缩