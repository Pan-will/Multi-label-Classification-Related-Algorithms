import tensorflow as tf

# 定义一个变量用于计算滑动平均，初始值为 0 ，类型为实数
v1 = tf.Variable(0, dtype=tf.float32)
# step变量用来模拟神经网络中迭代的轮数，即所谓的num_updates参数，用来动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均类，初始化衰减率(0.99)和衰减率控制变量step
# 该函数返回一个ExponentialMovingAverage对象，该对象调用apply方法可以通过滑动平均模型来更新参数
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量滑动平均的操作。
# 这里的给定数据需要是列表的形式，每次执行这个操作时列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后变量的取值，此处输出为[0.0 , 0.0]
    # 初始化之后变量v1和v1的滑动平均都为0
    print("通过ema.average(v1)获取滑动平均之后变量的取值:", sess.run([v1, ema.average(v1)]))

    # 更新变量v1的值为5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值
    # 此时衰减率为min(0.99,(1+step)/(10+step)=0.1) = 0.1
    # 所以v1的滑动平均会被更新为0.1*0 + 0.9*5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 输出[5.0 , 4.5]

    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 计算v1的滑动平均值
    # 此时衰减率为min(0.99,(1+step)/(10+step)=0.999999) = 0.99
    # 所以v1的滑动平均会被更新为0.99*4.5 + 0.01*10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 输出[10.0, 4.5549998]

    # 再次更新滑动平均值，得到的新的滑动平均值为0.99*4.555 + 0.01*10 = 4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 输出[10.0, 4.6094499]
