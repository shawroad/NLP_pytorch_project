"""

@file  : model.py

@author: xiaolu

@time  : 2019-10-28

"""
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from utils import *


class CharCNN(nn.Module):
    def __init__(self, config):
        super(CharCNN, self).__init__()
        self.config = config

        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.vocab_size, out_channels=self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )   # (batch_size, num_channels, (max_len-6)/3)

        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )  # (batch_size, num_channels, (max_len-6-18)/(3*3))

        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        )  # (batch_size, num_channels, (max_len-6-18-18)/(3*3))

        conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        )  # (batch_size, num_channels, (max_len-6-18-18-18)/(3*3))

        conv5 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        )  # (batch_size, num_channels, (max_len-6-18-18-18-18)/(3*3))

        conv6 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )  # (batch_size, num_channels, (max_len-6-18-18-18-18-18)/(3*3*3))

        # Length of output after conv6
        conv_output_size = self.config.num_channels * ((self.config.max_len - 96) // 27)

        linear1 = nn.Sequential(
            nn.Linear(conv_output_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear2 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear3 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.output_size),
            nn.Softmax()
        )

        self.convolutional_layers = nn.Sequential(conv1, conv2, conv3, conv4, conv5, conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)

        # Initialize Weights 初始化权重
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, embedded_sent):
        embedded_sent = embedded_sent.transpose(1, 2)  # .permute(0,2,1) # shape=(batch_size,embed_size,max_len)
        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)  # 拉直
        linear_output = self.linear_layers(conv_out)
        return linear_output

    def add_optimizer(self, optimizer):
        '''
        添加优化器
        :param optimizer:
        :return:
        '''
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        '''
        添加损失函数
        :param loss_op:
        :return:
        '''
        self.loss_op = loss_op

    def reduce_lr(self):
        # 学习率递减
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        if epoch > 0 and epoch % 3 == 0:
            self.reduce_lr()  # 每3步学习率折半

        for i, batch in enumerate(train_iterator):
            _, n_true_label = batch
            if torch.cuda.is_available():
                batch = [Variable(record).cuda() for record in batch]
            else:
                batch = [Variable(record) for record in batch]

            x, y = batch

            self.optimizer.zero_grad()  # 梯度清零
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            self.eval()
            avg_train_loss = np.mean(losses)
            train_losses.append(avg_train_loss)

            val_accuracy = evaluate_model(self, val_iterator)
            print("epoch: %d, step: %d, loss: %f, accuracy: %f" %(epoch, i, avg_train_loss, val_accuracy))
            losses = []
            self.train()

        return train_losses, val_accuracies