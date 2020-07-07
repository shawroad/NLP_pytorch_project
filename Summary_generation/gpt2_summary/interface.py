# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 15:20
# @Author  : xiaolu
# @FileName: interface.py
# @Software: PyCharm

import torch
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join
import torch.nn.functional as F
from config import Config

PAD = '[PAD]'
pad_id = 0


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    '''
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    :param logits: logits distribution shape (vocabulary size)
    :param top_k: top_k > 0: keep only top k tokens with highest probability (top-k filtering).
    :param top_p: top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    :param filter_value:
    :return:
    '''
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    # 加载模型  词表
    model_path = join(Config.model_output_path, 'model_epoch{}'.format(1))

    tokenizer = BertTokenizer(vocab_file=Config.gpt2_vocab)

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(Config.device)
    model.eval()

    print('***********************Summary model start************************')
    max_len = 200  # 最多生成两百个token
    repetition_penalty = 1  # 惩罚项 主要是为了让生成的token在后面尽量少出现
    temperature = 1   # 控制样本的生成尽可能多样性
    topk = 8  # 最高k选1
    topp = 0   # 最高累计概率

    while True:
        try:
            text = input()  # 输入一篇新闻
            for i in range(5):
                if len(text):
                    text = text[:1000]
                input_ids = [tokenizer.cls_token_id]
                input_ids.extend(tokenizer.encode(text))
                input_ids.append(tokenizer.sep_token_id)
                curr_input_tensor = torch.tensor(input_ids).long().to(Config.device)

                generated = []
                # 最多生成max_len个token
                for _ in range(max_len):
                    outputs = model(input_ids=curr_input_tensor)
                    next_token_logits = outputs[0][-1, :]
                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率

                    for id in set(generated):
                        next_token_logits[id] /= repetition_penalty

                    next_token_logits = next_token_logits / temperature

                    # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                    next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')

                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)

                    # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                    if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                        break

                    generated.append(next_token.item())
                    curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)

                text = tokenizer.convert_ids_to_tokens(generated)
                print("summary:" + "".join(text))

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
