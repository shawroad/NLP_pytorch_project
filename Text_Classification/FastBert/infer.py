"""

@file  : infer.py

@author: xiaolu

@time  : 2020-05-29

"""

import argparse
import json
import time
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from model.model_fastbert import FastBertModel, BertConfig
from data_utils.dataset_preparing import PrepareDataset, TextCollate
from utils import load_json_config, init_bert_adam_optimizer, load_saved_model, save_model, eval_pr

# 随机数固定，RE-PRODUCIBLE
seed = 9999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

debug_break = False


def infer_model(model, dataset, num_workers=1, inference_speed=None, dump_info_file=None):
    global global_step
    global debug_break
    model.eval()
    infer_dataloader = data.DataLoader(dataset=dataset,
                                       collate_fn=TextCollate(dataset),
                                       batch_size=1,
                                       num_workers=num_workers,
                                       shuffle=False)
    correct_sum = 0
    num_sample = infer_dataloader.dataset.__len__()
    predicted_probs = []
    true_labels = []
    infos = []
    print("Inference Model...")
    cnt = 0
    stime_all = time.time()
    for step, batch in enumerate(tqdm(infer_dataloader, unit="batch", ncols=100, desc="Inference process: ")):
        texts = batch["texts"]
        tokens = batch["tokens"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        attn_masks = batch["attn_masks"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            probs, layer_idxes, uncertain_infos = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks,
                                                        inference=True, inference_speed=inference_speed)
        _, top_index = probs.topk(1)

        correct_sum += (top_index.view(-1) == labels).sum().item()
        cnt += 1
        if cnt == 1:
            stime = time.time()

        if dump_info_file is not None:
            for label, pred, prob, layer_i, text in zip(labels, top_index.view(-1), probs, [layer_idxes], texts):
                infos.append((label.item(), pred.item(), prob.cpu().numpy(), layer_i, text))
        if debug_break and step > 50:
            break

    time_per = (time.time() - stime) / (cnt - 1)
    time_all = time.time() - stime_all
    acc = format(correct_sum / num_sample, "0.4f")
    print("speed_arg:%s, time_per_record:%s, acc:%s, total_time:%s"%(inference_speed, format(time_per, '0.4f'), acc, format(time_all, '0.4f')))
    if dump_info_file is not None and len(dump_info_file) != 0:
        with open(dump_info_file, 'w') as fw:
            for label, pred, prob, layer_i, text in infos:
                fw.write('\t'.join([str(label), str(pred), str(layer_i), text]) + '\n')

    labels_pr = [info[0] for info in infos]
    preds_pr = [info[1] for info in infos]
    precise, recall = eval_pr(labels_pr, preds_pr)
    print("precise:%s, recall:%s"%(format(precise, '0.4f'), format(recall, '0.4f')))


def main(args):
    # 1. 加载配置文件
    config = load_json_config(args.model_config_file)

    # 2. 加载模型
    bert_config = BertConfig.from_json_file(config.get("bert_config_path"))
    model = FastBertModel(bert_config, config)
    load_saved_model(model, args.save_model_path)
    model = model.to(device)
    print('Initialize model Done'.center(60, '*'))

    # 3. 数据集的准备
    infer_dataset = PrepareDataset(vocab_file=config.get("vocab_file"),
                                   max_seq_len=config.get("max_seq_len"),
                                   num_class=config.get("num_class"),
                                   data_file=args.infer_data)

    print("Load INFER Dataset Done, Total eval line: ", infer_dataset.__len__())

    # 4. 开始infer
    infer_model(model, infer_dataset, num_workers=args.data_load_num_workers,
                inference_speed=args.inference_speed, dump_info_file=args.dump_info_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Textclassification training script arguments.")
    parser.add_argument("--model_config_file", dest="model_config_file", action="store",
                        help="The path of configuration json file.")

    parser.add_argument("--save_model_path", dest="save_model_path", action="store",
                        help="The path of trained checkpoint model.")

    parser.add_argument("--infer_data", dest="infer_data", action="store", help="")
    parser.add_argument("--dump_info_file", dest="dump_info_file", action="store", help="")

    parser.add_argument("--inference_speed", dest="inference_speed", action="store",
                        type=float, default=1.0, help="")

    # -1 for NO GPU
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0",
                        help="Device ids of used gpus, split by ',' , IF -1 then no gpu")

    parser.add_argument("--data_load_num_workers", dest="data_load_num_workers", action="store", type=int, default=1,
                        help="")
    parser.add_argument("--debug_break", dest="debug_break", action="store", type=int, default=0,
                        help="Running debug_break, 0 or 1.")

    parsed_args = parser.parse_args()
    # debug_break = (parsed_args.debug_break == 1)
    main(parsed_args)
