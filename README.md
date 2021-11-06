硕士小论文对比过的几个算法，有复现的、也有作者提供的，做个备份。场景：多标签数据分类，多标签预测。



## 搜罗了几个权威的多标签数据集下载网站（实验常用的数据集都能找到）：

1. http://waikato.github.io/meka/datasets/
2.  http://mulan.sourceforge.net/datasets-mlc.html
3. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
4. https://sci2s.ugr.es/keel/multilabel.php
5. https://www.uco.es/kdis/mllresources/
6. http://mlkd.csd.auth.gr/concept_drift.html
7. 带概念漂移的数据集：https://www.win.tue.nl/~mpechen/data/DriftSets/
8. 带冗余特征的数据集，用于特征选择实验：https://jundongl.github.io/scikit-feature/datasets.html



> 注：每个项目里有单独的readme.md文件做详细介绍。以下仅对每个算法作个概括。



## 8、增量式随机树和随机森林算法

如题。



## 7、MLDF

论文出处：Yang L , Wu X Z , Jiang Y , et al. Multi-Label Learning with Deep Forest[J]. 2019.

南京大学周志华教授团队的力作，是深度森林的变体及应用。

MLDF是多标签数据分类算法，模型是静态批量式学习。

实验中用到的数据集全都可以在  http://mulan.sourceforge.net/datasets-mlc.html  下载。



## 6、BatchMultiLabelClassifier

流场景、多标签分类算法，主要还是基于霍夫丁树的改编。

项目中有两种接收实例的方式：小batch形式，逐条接收形式。

项目中实现了有单棵霍夫丁树小批量、多棵霍夫丁树小批量、逐条增量式单棵霍夫丁树、逐条增量式霍夫丁森林。

特点：小batch形式接收数据，所以模型实际上还是批量学习。

可以跑。



## 5、Hybrid_Forest

来自论文：Hybrid Forest: A Concept Drift Aware Data Stream Mining Algorithm

模型可以跑，效果凑合，实验中用的对比算法太少了。



## 4、MLKNN

很牛逼的一个多标签分类算法，后面的什么级联模型、深度森林模型，在one-error、汉明损失、精度等几个指标 上，就没有比得过MLKNN的，KNN  yyds~。

项目里有MATLAB版和Python版。



## 3、Hoeffding Tree

基于决策树，可用作单标签、多标签数据分类和预测，项目中仅提供了实现霍夫丁树的算法，没有做具体的模型训练，也没有具体实验。可以拿来改编。



## 2、MyBlog

非数据分类算法，仅是学Django框架时写的一个Python版web项目，或许可用于本科课设。



## 1、NeuralNetworksDemos

非数据分类算法，仅是学神经网络初期写的几个小demo。比如MNIST数据分类，滑动平均模型。













