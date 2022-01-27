"""
@file   : focal_loss.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import torch
from torch import nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    # 可用于二分类和多标签分类
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        '''
        假设是三个标签的多分类
        loss_fct = BCEFocalLoss()
        labels = torch.tensor([[0, 1, 1], [1, 0, 1]])
        logits = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
        loss = loss_fct(logits, labels)
        print(loss)  # tensor(0.0908)
        '''
        probs = torch.sigmoid(logits)

        loss = -self.alpha * (1 - probs) ** self.gamma * labels * torch.log(probs) - (
                    1 - self.alpha) * probs ** self.gamma * (1 - labels) * torch.log(1 - probs)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class MultiCEFocalLoss(nn.Module):
    # 可以用于多分类 (注: 不是多标签分类)
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, logits, labels):
        '''
        logits: (batch_size, class_num)
        labels: (batch_size,)
        '''
        probs = F.softmax(logits, dim=1)
        class_mask = F.one_hot(labels, self.class_num)  # 将真实标签转为one-hot
        ids = labels.view(-1, 1)  # (batch_size, 1)
        alpha = self.alpha[ids.data.view(-1)]  # 每一类的权重因子

        probs = (probs * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    # loss_fct = BCEFocalLoss()
    # labels = torch.tensor([[0, 1, 1], [1, 0, 1]])
    # logits = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
    # loss = loss_fct(logits, labels)
    # print(loss)

    # 举例四分类
    loss_fct = MultiCEFocalLoss(class_num=4)
    labels = torch.tensor([1, 3, 0, 0, 2])
    logits = torch.randn(5, 4)
    loss = loss_fct(logits, labels)
    print(loss)