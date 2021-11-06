import numpy as np


def mlknn(train, test, id, label_columns, k):
    smooth = 1.0
    # 计算每个标签出现的概率
    phj = {}
    for label in label_columns:
        phj[label] = (smooth + train[train[label] == 1].shape[0]) / (smooth * 2 + train.shape[0])

    train_ids = train[id].values
    tmp_train = train.drop(label_columns + [id], axis=1)
    test_ids = test[id].values
    test_labels = test[label_columns]
    tmp_test = test.drop(label_columns + [id], axis=1)
    data_columns = tmp_train.columns

    # 计算训练集每个样本之间的相似度，并保存跟每个样本最相似的K个样本
    knn_records_train = {}
    cos_train = {}
    for i in range(tmp_train.shape[0]):
        record = tmp_train.iloc[i]
        norm = np.linalg.norm(record)
        cos_train[train_ids[i]] = {}

        for j in range(tmp_train.shape[0]):
            if cos_train.has_key(train_ids[j]) and cos_train[train_ids[j]].has_key(train_ids[i]):
                cos_train[train_ids[i]][train_ids[j]] = cos_train[train_ids[j]][train_ids[i]]
            else:
                cos = np.dot(record, tmp_train.iloc[j]) / (norm * np.linalg.norm(tmp_train.iloc[j]))
                cos_train[train_ids[i]][train_ids[j]] = cos
        topk = sorted(cos_train[train_ids[i]].items(), key=lambda item: item[1], reverse=True)[0:k]
        knn_records_train[train_ids[i]] = [item[0] for item in topk]

    kjr = {}
    not_kjr = {}
    for label in label_columns:
        kjr[label] = {}
        not_kjr[label] = {}
        for m in range(train.shape[0]):
            record = train.iloc[m]
            if record[label] == 1:
                # 计算标签为1并且相邻K个样本中标签也为1的样本个数
                r = 0
                for rec_id in knn_records_train[train_ids[m]]:
                    if train[train[id] == rec_id][label].values[0] == 1:
                        r += 1
                if not kjr[label].has_key(r):
                    kjr[label][r] = 1
                else:
                    kjr[label][r] += 1
            else:
                # 计算标签为0并且相邻K个样本中标签也为1的样本个数
                r = 0
                for rec_id in knn_records_train[train_ids[m]]:
                    if train[train[id] == rec_id][label].values[0] == 1:
                        r += 1
                if not not_kjr[label].has_key(r):
                    not_kjr[label][r] = 1
                else:
                    not_kjr[label][r] += 1

    # 计算当前样本标签为1条件下，K个近邻样本中标签为1个数为Cj的概率
    pcjhj = {}
    for label in label_columns:
        pcjhj[label] = {}
        for L in range(k + 1):
            if kjr[label].has_key(L):
                pcjhj[label][L] = (smooth + kjr[label][L]) / (smooth * (k + 1) + sum(kjr[label].values()))
            else:
                pcjhj[label][L] = (smooth + 0) / (smooth * (k + 1) + sum(kjr[label].values()))

    # 计算当前样本标签为0条件下，K个近邻样本中标签为1个数为Cj的概率
    not_pcjhj = {}
    for label in label_columns:
        not_pcjhj[label] = {}
        for L in range(k + 1):
            if not_kjr[label].has_key(L):
                not_pcjhj[label][L] = (smooth + not_kjr[label][L]) / (smooth * (k + 1) + sum(not_kjr[label].values()))
            else:
                not_pcjhj[label][L] = (smooth + 0) / (smooth * (k + 1) + sum(not_kjr[label].values()))

    # 计算测试集中每个样本与训练集样本之间的相似度，并保存跟每个样本最相似的K个样本
    knn_records_test = {}
    cos_test = {}
    for i in range(tmp_test.shape[0]):
        record = tmp_test.iloc[i]
        norm = np.linalg.norm(record)
        cos_test[test_ids[i]] = {}

        for j in range(tmp_train.shape[0]):
            cos = np.dot(record, tmp_train.iloc[j]) / (norm * np.linalg.norm(tmp_train.iloc[j]))
            cos_test[test_ids[i]][train_ids[j]] = cos
        topk = sorted(cos_test[test_ids[i]].items(), key=lambda item: item[1], reverse=True)[0:k]
        knn_records_test[test_ids[i]] = [item[0] for item in topk]

    pred_test_labels = {}
    correct_rec = 0
    for i in range(tmp_test.shape[0]):
        record = tmp_test.iloc[i]
        correct_col = 0
        for label in label_columns:
            if not pred_test_labels.has_key(label):
                pred_test_labels[label] = []
            # 计算每个测试样本K近邻中标签为1的个数
            cj = 0
            for rec_id in knn_records_test[test_ids[i]]:
                if train[train[id] == rec_id][label].values[0] == 1:
                    cj += 1
            # 计算包含Cj个标签为1的K近邻条件下，该测试样本标签为1的概率
            phjcj = phj[label] * pcjhj[label][cj]
            # 计算包含Cj个标签为1的K近邻条件下，该测试样本标签为0的概率
            not_phjcj = (1 - phj[label]) * not_pcjhj[label][cj]

            if phjcj > not_phjcj:
                pred_test_labels[label].append(1)
                pred_label = 1
            else:
                pred_test_labels[label].append(0)
                pred_label = 0
            if pred_label == test_labels[label].values[i]:
                correct_col += 1
        if correct_col == len(label_columns):
            correct_rec += 1
    print('测试集标签识别准确率', correct_rec * 1.0 / test.shape[0])


if __name__ == '__main__':
    s = mlknn()
