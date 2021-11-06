"""
假设我们要最小化函数 y=x^2, 选择初始点x0=5
"""
# 1. 学习率为1的时候，x在5和-5之间震荡。
import tensorflow as tf

# 迭代次数
TRAINING_STEPS = 10
# 学习率设为1，那么参数的更新幅度太大，导致来回震荡。
LEARNING_RATE = 1
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        x_value = sess.run(x)
        print("在第 %s 次迭代后: x%s 变成 %f." % (i + 1, i + 1, x_value))

print("\n*************** 二 ***************")
# 2. 学习率为0.001的时候，下降速度过慢，在901轮时才收敛到0.823355。
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 100 == 0:
            x_value = sess.run(x)
            print("在第 %s 次迭代后: x%s 变成 %f." % (i + 1, i + 1, x_value))

print("\n*************** 三 ***************")
# 3. 学习率为0.09时，算是一个比较正常的学习率
TRAINING_STEPS = 1000
LEARNING_RATE = 0.009
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 100 == 0:
            x_value = sess.run(x)
            print("在第 %s 次迭代后: x%s 变成 %f." % (i + 1, i + 1, x_value))
print("\n*************** 四 ***************")

# 4. 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得不错的收敛程度。
TRAINING_STEPS = 100
global_step = tf.Variable(0)
# 通过exponential_decay函数生成学习率
# 初始学习率为0.1，指定了staircase=True，所以没训练100轮后，学习率乘以0.96
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
# 使用指数衰减的学习率。在minimize函数中传入global_step将自动更新global_step参数，从而跟新学习率
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 10 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print(
                "在第 %s 次迭代后: x%s 变成 %f, 学习率是 %f." % (i + 1, i + 1, x_value, LEARNING_RATE_value))
