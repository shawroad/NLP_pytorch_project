1. 再训练，或者重新预训练(在你垂直领域的数据集上)。首先将你的数据放到data目录下, 使用data_process.py进行整理。
    然后，执行get_train_data.py得到bert的输入数据。最后运行run_pretrain_bert.py开启预训练。
2. 当预训练结束后，将再训练得到的模型放入retrain_bert目录下，执行bert_corrector.py即可纠错。

样例: 
   origin_error_text: "产拳制度不够明悉，政腐作为自然资元掌握着"
   output_correct_text: "产权制度不够明熟，政权作为自然资源掌握着"