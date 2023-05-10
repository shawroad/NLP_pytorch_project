# encoding: GBK
import torch
import json
from tqdm import tqdm
import time
import os
import argparse
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/alpaca_data_zh_51k.jsonl', type=str, help='')
    parser.add_argument('--pretrained_model', default='/root/autodl-tmp/chatglm/chatglm_pretrain', type=str, help='')
    parser.add_argument('--lora_model', default='/root/autodl-tmp/chatglm/my_chatglm_lora/output/global_step-1001', type=str, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--max_src_len', type=int, default=128, help='')
    return parser.parse_args()


def main():
    args = set_args()
        
    model = ChatGLMForConditionalGeneration.from_pretrained(args.pretrained_model)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.lora_model)
        
    model = PeftModel.from_pretrained(model, args.lora_model, torch_dtype=torch.float32)
    model.half().cuda()
    
    # model.eval()
    save_data = []
    f1 = 0.0
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()
    
    with open(args.test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[2:]:
            sample = json.loads(line.strip())
            src_tokens = tokenizer.tokenize(sample['context'])
            if len(src_tokens) > args.max_src_len:
                src_tokens = src_tokens[:args.max_src_len]
                        
            tokens = src_tokens + ['[gMASK]', '<sop>']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
            input_ids = torch.tensor([input_ids]).cuda()
            generation_kwargs = {
                "min_length": 5,
                "max_new_tokens": max_tgt_len,
                "top_p": 0.7,
                "temperature": 0.95,
                "do_sample": False,
                "num_return_sequences": 1,
            }
    
            response = model.stream_generate(input_ids, **generation_kwargs)
            for s in response:
                # tensor([[ 57010,  20012,  20005,  84432, 127502,  86263,  86231,  83860,  84651, 116818,  20031,  20004,  33049,  20012, 150001, 150004,  20005]], device='cuda:0')
                ids_list = s.tolist()[0] 
                res = tokenizer.decode(ids_list).replace('<eop>', '')
                print(res)
                time.sleep(1)
            exit()
            response = model.generate_one(input_ids, **generation_kwargs)

            res = []
                    
            for i_r in range(generation_kwargs["num_return_sequences"]):
                outputs = response.tolist()[i_r][input_ids.shape[1]:]
                r = tokenizer.decode(outputs).replace("<eop>", "")
                res.append(r)
            pre_res = [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
            real_res = sample["target"].split("\n")
            print('-'*50)
            print(real_res)
            print('*'*100)
            
            same_res = set(pre_res) & set(real_res)
            if len(set(pre_res)) == 0:
                p = 0.0
            else:
                p = len(same_res) / len(set(pre_res))
            r = len(same_res) / len(set(real_res))
            if (p + r) != 0.0:
                f = 2 * p * r / (p + r)
            else:
                f = 0.0
            f1 += f
            save_data.append({"context": sample["context"], "ori_answer": sample["target"], "gen_answer": res[0], "f1": f})
    
    save_path = os.path.join(args.model_dir, "ft_pt_answer.json")
    json.dump(save_data, open(save_path, 'w', encoding='utf8'), ensure_ascii=False)
    '''
    e_time = time.time()
    # print("time cost£º{}s".format(e_time - s_time))
        
    print(f1 / 50)
    save_path = os.path.join(args.model_dir, "ft_pt_answer.json")
    fin = open(save_path, "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)
    fin.close()
    '''


if __name__ == '__main__':
    main()
