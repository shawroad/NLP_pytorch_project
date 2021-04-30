"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-30
"""
import argparse


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./GPT2_config/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--vocab_path', default='./GPT2_config/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')

    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--title_max_len', type=int, default=32, help='生成标题的最大长度，要比max_len小')
    parser.add_argument('--train_file_path', default='./data/train_data.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='./data/test_data.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--data_dir', default='./data', type=str, help='生成缓存数据的存放路径')

    parser.add_argument('--num_train_epochs', default=5, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='梯度积累')

    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=4000, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')


    return parser.parse_args()
