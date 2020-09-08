1. 预处理数据，在corpus下有个corpus.txt文件，一行代表一篇文章，然后执行pro_data.py可以得到pro_data.txt文件，相当于
   将文章进行按句分开，然后每篇文章用空一行的方式隔开
2. 这里我们用bert-base的配置config和词表vocab.txt
3. 执行get_train_data.py 得到训练数据，这里包含token转id 以及对句子进行mask
4. python run_pretrain.py 开始预训练