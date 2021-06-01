"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-27
"""
import os
import pickle
import random
import timeit
import torch
import json
import numpy as np
from tqdm import tqdm
from config import set_args
from model import Model
from adversarial import FGM, PGD
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from utils import MyProcessor, squad_convert_examples_to_features_orig, SquadResult
from metrics import compute_predictions_logits, squad_evaluate, baidu_evaluate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, prefix="dev", step=0):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, set_type=prefix, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(dataset)   # 对数据采样 这里选用全部的数据
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = {}".format(len(dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }

            example_indices = batch[3]   # 样本索引
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):   # 遍历当前batch中的每个样本的原始样本
            eval_feature = features[example_index.item()]   # 取出当前样本在那个原始样本
            unique_id = int(eval_feature.unique_id)   # 当前样本的id
            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output[:2]
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    if prefix == 'test':
        with open(os.path.join(args.output_dir, args.test_prob_file), 'wb') as f:
            pickle.dump(all_results, f)

    evalTime = timeit.default_timer() - start_time
    print('整个验证过程的总时间: {:10f}, 平均每个批次的时间:{:10f}'.format(evalTime, evalTime / len(dataset)))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}_{}.json".format(prefix, step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}.json".format(prefix, step))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}_{}.json".format(prefix, step))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=all_results,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        do_lower_case=args.do_lower_case,
        output_prediction_file=output_prediction_file,
        output_nbest_file=output_nbest_file,
        output_null_log_odds_file=output_null_log_odds_file,
        verbose_logging=args.verbose_logging,
        version_2_with_negative=args.version_2_with_negative,
        null_score_diff_threshold=args.null_score_diff_threshold,
        tokenizer=tokenizer
    )
    if prefix == 'dev':
        results = squad_evaluate(examples, predictions)
        return results
    else:
        return None


def train(args, train_dataset, model, tokenizer):
    # 选择用哪一种对抗训练
    if args.do_fgm:
        fgm = FGM(model)
    if args.do_pgd:
        pgd = PGD(model, k=3)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    warmup_steps = int(t_total * args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

    # 学习率线性衰减
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    set_seed(args)

    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Num batch_size = {}".format(args.train_batch_size))
    print("  Num gradient_accumulation_steps = {}".format(args.gradient_accumulation_steps))
    print("  Num total step = {}".format(t_total))

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    for epoch in range(args.num_train_epochs):
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)   # 将数据搬到gpu
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'start_positions': batch[3],
                'end_positions': batch[4]
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            print('epoch:{}, cur_epoch_step:{}, global_step:{}, loss:{}'.format(epoch, step, global_step, loss))

            tr_loss += loss.detach().item()

            if args.do_fgm:
                # fgm对抗训练
                fgm.attack()
                outputs_adv = model(**inputs)
                loss_adv = outputs_adv[0]
                loss_adv = loss_adv.mean() / args.gradient_accumulation_steps
                loss_adv.backward()
                fgm.restore()

            if args.do_pgd:
                # 对抗训练-PGD
                pgd.backup_grad()
                for t in range(pgd.k):
                    pgd.attack(is_first_attack=(t == 0))  # 在 embedding 上添加对抗扰动, first attack 时备份 param.data
                    if t != pgd.k-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    outputs_adv = model(**inputs)
                    loss_adv = outputs_adv[0]
                    loss_adv = loss_adv.mean() / args.gradient_accumulation_steps
                    loss_adv.backward()   # 反向传播，并在正常的 grad 基础上，累加对抗训练的梯度
                pgd.restore()   # 恢复 embedding 参数

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                results = evaluate(args, model, tokenizer, prefix='dev', step=global_step)
                json.dump(results, open('result.json', 'a+', encoding='utf8'))
                
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if args.max_steps > 0 and global_step > args.max_steps:
                print('达到了最大的步数，终止训练咯!!!')
                break
    return global_step, tr_loss / global_step


def load_and_cache_examples(args, tokenizer, set_type='train', output_examples=False):
    '''
    预处理数据+加载数据
    output_examples: 预处理后的数据是否进行保存
    is_evaluate: 是否是验证数据集
    '''

    # 1. 生成处理数据保存的路径
    input_dir = args.feature_dir if args.feature_dir else "."
    cached_features_file = os.path.join(input_dir, "cached_{}_{}".format(set_type, str(args.max_seq_length)))

    # 2. 看那个路径在不在   在的话  说明数据已处理  直接加载就完事了 否则 得预处理数据
    if os.path.exists(cached_features_file) and not args.overwrite_cache:   # overwrite_cache是否重新预处理
        print('Loading features from cached file {}'.format(cached_features_file))
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )

    else:
        print('Creating features form dataset file at {}'.format(input_dir))
        processor = MyProcessor()
        examples = ''
        if set_type == 'dev':
            examples = processor.get_dev_examples(filename=args.eval_file)
        elif set_type == 'train':
            examples = processor.get_train_examples(filename=args.train_file)
        elif set_type == 'test':
            examples = processor.get_test_examples(filename=args.test_file)

        features, dataset = squad_convert_examples_to_features_orig(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=set_type == 'train',
        )
        # 保存预处理后的数据集
        torch.save({'features': features, 'dataset': dataset, 'examples': examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    args = set_args()

    tokenizer = BertTokenizer.from_pretrained(args.vocab, do_lower_case=args.do_lower_case)

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, set_type='train', output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        print(" global_step = {}, average loss = {}".format(global_step, tr_loss))


if __name__ == "__main__":
    main()
