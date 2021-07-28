"""
@file   : run_bert_whitening.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-28$
"""
import os
import torch
import numpy as np
import scipy.stats
from tqdm import tqdm
from config import set_args
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel
from utils import load_STS_data, transform_and_normalize, compute_kernel_bias, save_whiten, load_whiten


def sent_to_vec(sent, tokenizer, model, pooling, max_length):
    with torch.no_grad():
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        if torch.cuda.is_available():
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['token_type_ids'] = inputs['token_type_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()

        hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states
        # print(len(hidden_states))   # 13  1层embedding + 12层transformer-encoder

        if pooling == 'first_last':
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'last_avg':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif pooling == 'last2avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif pooling == 'cls':
            output_hidden_state = (hidden_states[-1])[:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(args.pooling))

        vec = output_hidden_state.cpu().numpy()[0]
    return vec


def sents_to_vecs(sents, tokenizer, model, pooling, max_length):
    vecs = []
    for sent in tqdm(sents, total=len(sents)):
        vec = sent_to_vec(sent, tokenizer, model, pooling, max_length)
        # print(len(vec))   # 768 长度768的向量
        vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def evaluate(test_data, model):
    label_list = [int(x[2]) for x in test_data]
    label_list = np.array(label_list)    # 标签整成ndarry

    sent1_embeddings, sent2_embeddings = [], []
    for sent in tqdm(test_data, total=len(test_data), desc="get sentence embeddings!"):
        vec = sent_to_vec(sent[0], tokenizer, model, args.pooling, args.max_len)
        sent1_embeddings.append(vec)
        vec = sent_to_vec(sent[1], tokenizer, model, args.pooling, args.max_len)
        sent2_embeddings.append(vec)
    target_embeddings = np.vstack(sent1_embeddings)
    target_embeddings = transform_and_normalize(target_embeddings, kernel, bias)  # whitening
    source_embeddings = np.vstack(sent2_embeddings)
    source_embeddings = transform_and_normalize(source_embeddings, kernel, bias)  # whitening

    similarity_list = F.cosine_similarity(torch.Tensor(target_embeddings),
                                          torch.tensor(source_embeddings))
    similarity_list = similarity_list.cpu().numpy()
    corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef


if __name__ == '__main__':
    args = set_args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path)
    if torch.cuda.is_available():
        model.cuda()

    dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
    test_data = load_STS_data("./data/STS-B/cnsd-sts-test.txt")

    output_filename = "{}-whiten.pkl".format(args.pooling)
    output_path = os.path.join(args.save_path, output_filename)
    if not os.path.exists(output_path):
        train_data = load_STS_data("./data/STS-B/cnsd-sts-train.txt")
        sents = [x[0] for x in train_data] + [x[1] for x in train_data]   # 将所有的句子拿出来  句子列表

        print("Transfer sentences to BERT embedding vectors.")   # 将训练集的所有句子转为向量
        vecs_train = sents_to_vecs(sents, tokenizer, model, args.pooling, args.max_len)

        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([vecs_train])
        save_whiten(output_path, kernel, bias)
    else:
        kernel, bias = load_whiten(output_path)
        import os
        

    corrcoef = evaluate(dev_data, model)
    print("dev corrcoef: {}".format(corrcoef))
    corrcoef = evaluate(test_data, model)
    print("test corrcoef: {}".format(corrcoef))


