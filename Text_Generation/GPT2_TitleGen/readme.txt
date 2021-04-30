1. 准备数据集 放在./data目录下。  分别为train_text.txt(一行一篇文章)、train_label.txt(一行一个标题)
2. 运行clean_data.py对数据进行清洗， python clean_data.py
3. 训练  python train.py   若要修改模型的层数之类的东西，在GPT2_config/config.json下进行修改
4. 推理  python inference.py