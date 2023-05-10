# encoding: GBK
import torch
import json
from tqdm import tqdm
import time
import os
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
import flask
import requests
from flask import Flask, render_template, request
from loguru import logger


logger.add('./logger/log_rizhi.log')

app = Flask("ChatPUA")


@app.route('/')
def index():
    return render_template('chatgpt_clone.html')


@app.route('/chatgpt-clone', methods=['POST', "GET"])
def chatgpt_clone():
    question = request.args.get('question', '')

    logger.info('问题:{}'.format(question))
    question = str(question).strip()
    if len(question) > 0:
        def stream():
            response = gen_answer(question)
            for s in response:
                if s == 'stop':
                    logger.info('答案:{}'.format(data))
                    data = '[DONE]'
                else:
                    ids_list = s.tolist()[0]
                    data = tokenizer.decode(ids_list).replace('<eop>', '')
                yield "data: %s\n\n" % data.replace('\n', '<br />').replace(question, '')
        return flask.Response(stream(), mimetype="text/event-stream")
    else:
        return '没有内容'




path = '/root/autodl-tmp/chatglm/chatglm_pretrain'
model = ChatGLMForConditionalGeneration.from_pretrained(path)
tokenizer = ChatGLMTokenizer.from_pretrained(path)
lora_model_path = '/root/autodl-tmp/chatglm/my_chatglm_lora/output/global_step-2000'
model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype=torch.float32)
model.half().cuda()


def gen_answer(question):
    max_len = 512
    max_src_len = 128
    max_tgt_len = max_len - max_src_len - 3
    src_tokens = tokenizer.tokenize(question)
    if len(src_tokens) > max_src_len:
        src_tokens = src_tokens[:max_src_len]
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
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    response = model.stream_generate(input_ids, **generation_kwargs)
    return response
    

# app.run(port='6006')
