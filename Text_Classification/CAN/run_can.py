"""
@file   : run_can.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-11-08
"""
import numpy as np


def get_calc_rate():
    # 从训练集统计先验分布   即每个标签的分布
    prior = np.zeros(num_classes)
    for d in train_data:
        prior[d[1]] += 1.
    prior /= prior.sum()
    return prior

if __name__ == '__main__':
    # 预测结果，计算修正前准确率
    y_pred = np.array([
        [0.2, 0.5, 0.2, 0.1],
        [0.3, 0.1, 0.5, 0.1],
        [0.4, 0.1, 0.1, 0.4],
        [0.1, 0.1, 0.1, 0.8],
        [0.3, 0.2, 0.2, 0.3],
        [0.2, 0.2, 0.2, 0.4]
    ])
    num_classes = y_pred.shape[1]    # 类别数
    y_true = np.array([0, 1, 2, 3, 1, 2])   # 真实的标签
    acc_original = np.mean([y_pred.argmax(axis=1) == y_true])
    print('original acc: %s' % acc_original)

    # prior = get_calc_rate()
    prior = np.array([0.2, 0.2, 0.25, 0.35])   # 这里我们随机设

    # 评价每个预测结果的不确定性
    k = 3   # 计算top-k熵
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)   # 对每个样本的logits归一化
    y_pred_entropy = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)  # 计算每一个样本的top-k熵
    print(y_pred_entropy)

    # 选择阈值，划分高、低置信度两部分
    threshold = 0.9
    y_pred_confident = y_pred[y_pred_entropy < threshold]  # top-k熵低于阈值的是高置信度样本
    y_pred_unconfident = y_pred[y_pred_entropy >= threshold]   # top-k熵高于阈值的是低置信度样本
    y_true_confident = y_true[y_pred_entropy < threshold]    # 取出高置信样本的真实标签
    y_true_unconfident = y_true[y_pred_entropy >= threshold]    # 取出低置信样本的真实标签

    # 显示两部分各自的准确率
    # 一般而言，高置信度集准确率会远高于低置信度的
    acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
    acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
    print('confident acc: %s' % acc_confident)
    print('unconfident acc: %s' % acc_unconfident)

    # 逐个修改低置信度样本，并重新评价准确率
    right, alpha, iters = 0, 1, 1  # 正确的个数，alpha次方，iters迭代次数
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0) # Y is L_0
        for _ in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)   # 列归一化  (4, 4)
            Y *= prior[None]
            # print(prior[None].shape)   # (1, 4)
            # print(Y.shape)    # (4, 4)
            Y /= Y.sum(axis=1, keepdims=True)    # 行归一化
        y = Y[-1]   # 取出调整的那个样本
        if y.argmax() == y_true_unconfident[i]:
            right += 1

    # 输出修正后的准确率
    acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
    print('new unconfident acc: %s' % (right / (i + 1.)))
    print('final acc: %s' % acc_final)