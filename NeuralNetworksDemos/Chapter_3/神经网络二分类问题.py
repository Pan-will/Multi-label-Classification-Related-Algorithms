import tensorflow as tf
from numpy.random import RandomState

"""一、定义神经网络的参数，输入和输出节点"""
batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义placeholder作为存放输入数据的地方。
# 维度不是固定的。若维度确定，则给出维度可以降低出错的概率。
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

print(type(w1), type(w2), type(x), type(y_))

"""二、定义前向传播过程，损失函数及反向传播算法"""
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 使用sigmoid()函数将y转换为0~1之间的数值。转换后y代表测试是正样本的概率，1-y是负样本的概率；
y = tf.sigmoid(y)
# 用交叉熵作为损失函数，来表示预测值与真实值之间的差距
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

"""三、生成模拟数据集"""
# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签。
# 这里所有x1 + x2 < 1的样例被认为是正样本；其他是负样本；
# 具体的：1表示正样本，0表示负样本；
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

"""四、创建一个会话来运行TensorFlow程序"""
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)

    # 输出训练后的参数值
    print("\n四、输出训练后的参数值:\n")
    print("w1=", sess.run(w1))
    print("w2=", sess.run(w2))

    """五、设定训练轮数"""
    STEPS = 500
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        # 每隔一段时间计算在所有数据上的交叉熵并输出
        if i % 100 == 0:
            total_aross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("%d轮训练后,loss为：%g " % (i, total_aross_entropy))

    print("\n五、500轮训练后的参数值:\n")
    print("w1=", sess.run(w1))
    print("w2=", sess.run(w2))
