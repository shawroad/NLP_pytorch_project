"""
@file   : run_customized_by_oneself_lora.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-04-21
"""
from transformers.models.bert import BertForSequenceClassification, BertTokenizer
from peft import LoraConfig, get_peft_model

model = BertForSequenceClassification.from_pretrained('./mengzi_pretrain')

# # 查看当前网络中都有哪些模块
# for x in model.modules():   # 查看模块
#     print(x)

# # 准备lora
# 计划在那个模块上应用lora, lora内部的代码是通过正则匹配的    可参考LoraConfig提供的写法。
TARGET_MODULES = [
    "query",
    "key",
]

peft_config = LoraConfig(task_type="SEQ_CLS", target_modules=TARGET_MODULES, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
exit()


