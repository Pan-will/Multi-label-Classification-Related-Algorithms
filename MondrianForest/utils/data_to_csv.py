import pandas as pd

# 读取data文件，指定属性，sep='[\s]*'意义为匹配一个或多个空格，因为原始数据集中数据分割是两个或者多个空格
data = pd.read_table('D:\Pycharm2020.1.3\WorkSpace\mondrianforest\dataset\waveform.data', header=None, sep=',')
# 生成csv文件
data.to_csv('D:\Pycharm2020.1.3\WorkSpace\mondrianforest\dataset\waveform.csv', index=False)
