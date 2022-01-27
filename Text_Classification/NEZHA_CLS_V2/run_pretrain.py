"""
@file   : run_pretrain.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-27
"""
import torch
from pdb import set_trace
from model.modeling_nezha import NeZhaModel, NeZhaConfig

if __name__ == '__main__':
    config = NeZhaConfig.from_pretrained('./nezha_pretrain/config.json')
    config.max_position_embeddings = 1024    # 可以随意指定最大长度
    config.output_hidden_states = True  # 输出所有的隐层
    config.output_attentions = True  # 输出所有注意力层计算结果
    nezha = NeZhaModel.from_pretrained('./nezha_pretrain', config=config)

    input_ids = torch.randint(0, 10000, size=(2, 732))
    output = nezha(input_ids=input_ids)
    # output[0].size()   # torch.Size([2, 732, 768])
    # output[1].size()   # torch.Size([2, 768]
    # len(output[2]), output[2][0].size()   # 13, torch.Size([2, 732, 768])
    # len(output[3]), output[3][0].size()     # 12,  torch.Size([2, 12, 732, 732])


    # 数据处理可以借鉴pretrain_model那个文件夹的处理方式。


