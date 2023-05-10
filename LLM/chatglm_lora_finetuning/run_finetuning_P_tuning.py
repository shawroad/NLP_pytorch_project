import torch
import deepspeed
import argparse
import os
from shutil import copy
from data_helper import GLMDataset, collate_fn 
from torch.utils.data import RandomSampler, DataLoader
from configuration_chatglm import ChatGLMConfig
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_epochs', default=20, type=int, help='')
    parser.add_argument('--train_batch_size', default=1, type=int, help='')
    parser.add_argument('--train_data_path', default='./data/train_corpus.json', type=str, help='')
    parser.add_argument('--pretrained_model', default="/root/autodl-tmp/chatglm/chatglm_pretrain", type=str, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--max_src_len', type=int, default=128, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')

    return parser.parse_args()


def main():
    args = set_args()

    config = ChatGLMConfig.from_pretrained(args.pretrained_model)
    config.pre_seq_len = args.pre_seq_len
    config.prefix_projection = args.prefix_projection

    model = ChatGLMForConditionalGeneration.from_pretrained(args.pretrained_model, config=config)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.pretrained_model)
    model = model.half().cuda()
    if args.prefix_projection:
        model.gradient_checkpointing_enable()

    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": args.log_steps
            }
    
    for name, param in model.named_parameters():
        if not any(nd in name for nd in ['prefix_encoder']):
            param.requires_grad = False
        else:
            # ['transformer.prefix_encoder.trans.0.weight', 'transformer.prefix_encoder.trans.2.bias', 'transformer.prefix_encoder.trans.2.weight', 'transformer.prefix_encoder.embedding.weight', 'transformer.prefix_encoder.trans.0.bias']
            print(name)  # train parameter
    
    print_trainable_parameters(model)
    # trainable params: 956600320 || all params: 7211806720 || trainable%: 13.264364356120737

    train_dataset = GLMDataset(args.train_data_path, tokenizer, args.max_len, args.max_src_len) 
    # train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  num_workers=0)

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()
    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels)
            loss = outputs[0]
            print('ChatGLM_Lora_Finetuning-epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            
            model_engine.backward(loss)
        
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if step == 1000:
                save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
                model_engine.save_pretrained(save_dir)
                copy(os.path.join(args.pretrained_model, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
                copy(os.path.join(args.pretrained_model, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))
            
    
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        model_engine.save_pretrained(save_dir)
        copy(os.path.join(args.pretrained_model, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
        copy(os.path.join(args.pretrained_model, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))


if __name__ == "__main__":
    main()
