"""

@file  : args.py

@author: xiaolu

@time  : 2020-03-03

"""
import torch


# 抽取出数据集的位置
search_input_file = "../data/extract/train/search.train.json"
zhidao_input_file = "../data/extract/train/zhidao.train.json"
dev_zhidao_input_file = "../data/extract/dev/zhidao.dev.json"
dev_search_input_file = "../data/extract/dev/search.dev.json"

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

seed = 42
max_seq_length = 512
max_query_length = 60
batch_size = 4
num_train_epochs = 4   # 训练多少个epoch
gradient_accumulation_steps = 8   # 梯度累积
# log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次


# output_dir = "./model_dir"
# predict_example_files = 'predict.data'
#
# max_para_num = 5  # 选择几篇文档进行预测
# learning_rate = 5e-5
# num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
