# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/13 15:06
@Auth ： xiaolu
@File ：tf2py.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""


import argparse
import os

import torch

from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
    load_tf_weights_in_xlnet,
)


GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


def convert_xlnet_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None
):
    # Initialise PyTorch model
    config = XLNetConfig.from_json_file(bert_config_file)

    finetuning_task = finetuning_task.lower() if finetuning_task is not None else ""
    if finetuning_task in GLUE_TASKS_NUM_LABELS:
        print("Building PyTorch XLNetForSequenceClassification model from configuration: {}".format(str(config)))
        config.finetuning_task = finetuning_task
        config.num_labels = GLUE_TASKS_NUM_LABELS[finetuning_task]
        model = XLNetForSequenceClassification(config)
    elif "squad" in finetuning_task:
        config.finetuning_task = finetuning_task
        model = XLNetForQuestionAnswering(config)
    else:
        model = XLNetLMHeadModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)

    # Save pytorch-model
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
    print("Save PyTorch model to {}".format(os.path.abspath(pytorch_weights_dump_path)))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print("Save configuration file to {}".format(os.path.abspath(pytorch_config_dump_path)))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default='xlnet_model.ckpt', type=str, required=False, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--xlnet_config_file",
        default='xlnet_config.json',
        type=str,
        required=False,
        help="The config json file corresponding to the pre-trained XLNet model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default='',
        type=str,
        required=False,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    parser.add_argument(
        "--finetuning_task",
        default=None,
        type=str,
        help="Name of a task on which the XLNet TensorFlow model was fine-tuned",
    )
    args = parser.parse_args()
    print(args)

    convert_xlnet_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.xlnet_config_file, args.pytorch_dump_folder_path, args.finetuning_task
    )