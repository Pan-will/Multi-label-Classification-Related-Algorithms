import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题，所以只有一个节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
# 定义了一个单层的神经网络前向传播的过程，此处就是加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数
# 定义预测多了、少了的成本：少了就少挣10块，多了才少挣1块；所以预测少了的损失大，于是模型应该偏向多的方向预测
loss_less = 10
loss_more = 1
#
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 设置回归的正确值为两个输入的和加上一个随机量。
# 加随机量是为了加入不可预测的噪音，否则测试不出不同损失函数的区别。
# 因为不同损失函数都会在能完全预测正确的时候最低。
# 一般情况噪音是一个均值为0的小量，所以这里设置噪音为-0.05~0.05的随机数。
Y = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print("经过 %d 轮训练后，w1是: " % (i))
            print(sess.run(w1), "\n")
    print("w1的最终结果是：\n", sess.run(w1))
    x1, x2 = sess.run(w1)[0], sess.run(w1)[1]
    print("所得预测函数表达式为：y=%.2fa+%.2fb" % (x1, x2))
