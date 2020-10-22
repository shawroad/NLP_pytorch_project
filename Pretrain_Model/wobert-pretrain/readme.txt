1. 将一篇文章按句子分开  一句一行，不同文章用空格隔开。
2. 运行process_pretrain_data.py文件，生成被mask数据（训练数据）, 若想进行动态mask  则运行多次，保存在不同地方，生成多种mask的数据
3. 运行 run_pretrain.py