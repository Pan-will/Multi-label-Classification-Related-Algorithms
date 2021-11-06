# Multi-label-Classification-Related-Algorithms
硕士小论文对比过的几个算法，有复现的、也有作者提供的，做个备份。场景：多标签数据分类，多标签预测。



> 注：每个项目里有单独的readme.md文件做详细介绍。以下仅对每个算法作个概括。



## 1、Hoeffding Tree

基于决策树，可用作单标签、多标签数据分类和预测，项目中仅提供了实现霍夫丁树的算法，没有做具体的模型训练，也没有具体实验。可以拿来改编。



## 2、MyBlog

非数据分类算法，仅是学Django框架时写的一个Python版web项目，或许可用于本科课设。



## 3、NeuralNetworksDemos

非数据分类算法，仅是学神经网络初期写的几个小demo。比如MNIST数据分类，滑动平均模型。



## 4、Hybrid_Forest

来自论文：Hybrid Forest: A Concept Drift Aware Data Stream Mining Algorithm

模型可以跑，效果凑合，实验中用的对比算法太少了。



## 5、BatchMultiLabelClassifier

流场景、多标签分类算法，主要还是基于霍夫丁树的改编。

项目中有两种接收实例的方式：小batch形式，逐条接收形式。

项目中实现了有单棵霍夫丁树小批量、多棵霍夫丁树小批量、逐条增量式单棵霍夫丁树、逐条增量式霍夫丁森林。

特点：小batch形式接收数据，所以模型实际上还是批量学习。

可以跑。