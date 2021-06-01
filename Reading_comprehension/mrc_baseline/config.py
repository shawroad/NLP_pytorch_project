"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-27
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()

    # 数据预处理的参数
    parser.add_argument("--feature_dir", default='./data', type=str, help="数据处理完后保存的路径")
    parser.add_argument("--overwrite_cache", default=False, type=bool, help="是否重新预处理数据")
    parser.add_argument("--max_seq_length", default=512, type=int, help="模型最大输入长度")
    parser.add_argument("--max_query_length", default=32, type=int, help="问题的最大长度")
    parser.add_argument("--doc_stride", default=128, type=int, help="滑动窗口的最大步长")
    parser.add_argument("--train_file", default='./data/train_process.json', type=str, help="训练数据集")
    parser.add_argument("--eval_file", default='./data/dev_process.json', type=str, help="验证数据集")
    parser.add_argument("--test_file", default='./data/dev_process.json', type=str, help="测试数据集")
    parser.add_argument("--output_dir", default='./outputs', type=str, help='模型保存路径以及预测的所有结果保存的位置')

    # 预训练模型
    parser.add_argument("--pretrain_config", default='./bert_pretrain/bert_config.json', type=str, help='预训练模型的config')
    parser.add_argument("--pretrain_model", default='./bert_pretrain/pytorch_model.bin', type=str, help='预训练模型权重')
    parser.add_argument("--vocab", default='./bert_pretrain/vocab.txt', type=str, help='词表')
    parser.add_argument("--do_lower_case", default=True, type=bool, help="是否忽略大小写问题")

    # 运行参数
    parser.add_argument("--do_train", default=True, type=bool, help="是否训练")
    parser.add_argument("--do_eval", default=True, type=bool, help="是否验证")
    parser.add_argument("--do_test", default=True, type=bool, help="是否测试")
    parser.add_argument("--train_batch_size", default=2, type=int, help="训练的批次大小")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="验证的批次大小")
    parser.add_argument("--max_steps", default=0, type=int, help="可以指定总共训练多少步")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度积聚的步数",)
    parser.add_argument("--num_train_epochs", default=3, type=int, help="总共把数据训练几轮 和max_steps设置一个就行")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="热启动的步数占总体步数的比例")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率大小")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="是否使用权重衰减")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪")
    parser.add_argument("--logging_steps", default=1, type=float, help="多少步验证一下")
    parser.add_argument("--save_steps", default=10000, type=float, help="多少步保存一下模型")

    parser.add_argument("--model_name_or_path", default='./save_model', type=str, help="保存优化器的状态")
    parser.add_argument("--test_prob_file", default='test_prob_file.pkl', type=str, help="测试集预测出来的概率值")

    # prediction
    parser.add_argument("--n_best_size", default=10, type=int, help="每个文章预测多少个答案")
    parser.add_argument("--max_answer_length", default=32, type=int, help="预测答案的最大长度")
    parser.add_argument("--best_val_f1", type=float, default=0., help="best_val_f1")
    parser.add_argument("--best_val_step", type=int, default=0, help="best_val_step")

    # 两种对抗训练
    parser.add_argument("--do_fgm", default=False, type=bool, help="是否采用FGM对抗训练")
    parser.add_argument("--do_pgd", default=False, type=bool, help="是否使用PGD对抗训练")

    parser.add_argument("--version_2_with_negative", default=True, type=bool, help="是够包含不可回答问题")

    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    return args

