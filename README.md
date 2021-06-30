# NLP_pytorch_project

## Chatbot 

####  1. Bert_chatbot: 类似于UniLM的方式  
- python train.py   &emsp; # 训练代码  
- python infernece.py &emsp; # 模型的推理

#### 2. seq2seq_luong: 编码器为两层gru网络，解码器为一层gru网络，在编码器和解码器中间，加入了luong注意力。
- python train.py  &emsp; # 训练代码
- python inference.py   &emsp; # 模型的推理

#### 3. transformer_chatbot: 标准的transformer模型
- python train.py   &emsp; # 训练代码
- python chat.py  &emsp;  # 可以进行聊天 训练数据采用的是青云对话语料

## Distillation

#### 1. DynaBert: 华为的工作，主要是采用剪枝的方式对bert的某些结构进行裁剪。
- python train_teacher_model.py  &emsp; # 训练老师模型
- python train_tailor_model.py  &emsp; # 对老师模型剪枝

#### 2. rnn_distill_bert: 用一层lstm网络去蒸馏bert模型，只加入了软标签损失。
- python train_bert.py   &emsp; # 训练老师模型 bert
- python train_distill.py  &emsp;  # 蒸馏 用lstm学习bert的输出

#### 3. three_layer_self-attention_to_distill_bert: 看名字大概都知道啥意思了，就是写了三层transformer的encoder，然后去蒸馏bert模型
- python train_bert.py   &emsp; # 训练老师模型bert
- python train_distill.py  &emsp;  # 蒸馏

#### 4. tiny_bert: 华为的工作，tiny_bert的蒸馏方式是除了加入软标签损失，还加入了中间层的均方误差损失
- python train.py   &emsp; # 训练老师模型 bert
- python train_distill_v2.py   &emsp; # 蒸馏


## Embedding

#### 1. skipgram-word2vec: 使用skipgram的方式得到词向量
- python 001-skipgram-word2vec.py

#### 2. bert: 直接训练bert, 从头训练， 也可以使用此代码进行再训练
- python 002-bert.py

#### 3. albert: 直接训练albert，从头训练， 也可以使用此代码进行再训练
- python 003-albert.py

#### 4. NPLM: 传统方法
- python 004-NPLM.py

## NER

#### 1. Bert_CRF_Ner: bert模型加条件随机场进行序列标注任务
- python run_ner_crf.py   &emsp;  # 模型的训练
- python inference.py   &emsp;  # 模型的推理

#### 2. Bert_Softmax_Ner: 直接使用bert模型进行序列标注
- python train.py   &emsp;  # 模型的训练
- python inference.py   &emsp;  # 模型的推理

#### 3. BiLSTM_CRF_Ner: 使用双向的lstm网络和crf进行序列标注任务
- python train.py   &emsp;   # 模型的训练

## NMT

#### 1. GRU_attention: 编码器和解码器都是gru网络，中间加入一个普通的注意力机制(直接的加权和)
- python train.py   &emsp;  # 模型的训练

#### 2. Transformers_NMT: 标准的transformer结构做机器翻译
- python train.py   &emsp;  # 模型的训练

## Pretrain_Model

#### 1. bert-pretrain: bert模型的再训练，首先通过get_train_data.py进行数据预处理，包含15%的词进行mask等操作，然后在进行训练。
- python get_train_data.py   &emsp; # 数据预处理
- python run_pretrain.py   &emsp;   # 再训练

#### 2. wobert-pretrain: wobert的预训练模型由苏神给出，这里的再训练可以加入自己构建的词表，然后修改了bert的分词方式。
- python process_pretrain_data.py   &emsp;  # 数据预处理
- python run_pretrain.py   &emsp;   # 再训练


## Reading_comprehension
#### 1. BERT_MRC: 使用bert去做机器阅读理解任务。这里长预料采用直接阶段的方式。
- python train.py   &emsp;   # 模型训练

#### 2. BiDAF: 双向注意力流机制的机器阅读理解模型
- python data_process.py   &emsp;  # 首先对数据预处理
- python train_bidaf.py   &emsp;   # 模型的训练

#### 3. DocQA: 传统的模型
- python data_process.py   &emsp;  # 首先对数据预处理
- python train_DocQA.py   &emsp;   # 模型的训练

#### 4. Match_LSTM: 传统的模型，单纯的rnn结构。
- python data_process.py   &emsp;  # 首先对数据预处理
- python train_Match_Lstm.py   &emsp;   # 模型的训练

#### 5. QANet: 也是一个比较传统的模型，但是这个模型是第一个抛弃rnn结构的mrc模型，这个模型也是第一个将self-attention机制引入到mrc任务中的。
- python data_process.py   &emsp;  # 数据预处理
- python train.py   &emsp;   # 模型的训练

#### 6. RNet: 传统的模型
- python data_process.py   &emsp;  # 数据预处理
- python train_RNet.py   &emsp;   # 模型的训练

#### 7. Recurrence-hotpot-baseline: 首次产生用rnn结构处理多跳推理的问题，在hotpotqa数据集中，除了包含答案的预测，还有支撑事实的预测，以及相关段落的预测。
- python data_process.py   &emsp;  # 数据预处理
- python train.py   &emsp;   # 模型的训练

#### 8. albert_mrc: 使用albert预训练模型去做mrc任务
- python train_update.py   &emsp;   # 训练模型
- python inference.py   &emsp;   # 单条数据的推理
- python inference_all.py   &emsp;   # 所有数据的推理

#### 9. electra_bert: 使用electra预训练模型去做mrc任务
- python run_cail.py    &emsp;  # 训练模型
- python evaluate.py   &emsp;   # 模型的评估

#### 10. mrc_baseline: 如果做mrc任务，建议首先看这个代码，这里面包含了mrc注意的各种细节，如长文本的处理(滑动窗口)， 答案的排序，对抗训练等等。
- python train.py   &emsp;   # 训练模型

#### 11. roberta_mrc: 使用roberta预训练模型去做mrc任务
- python train.py   &emsp;   # 训练模型

#### 12. transformer+rnn+attention: 这个项目是做生成式的阅读理解，直接采用的是seq2seq的方式，编码器采用的是transformer的encoder, 解码器采用的gru的结构，中间还加入了一层普通的注意力机制。
- python train.py   &emsp;   # 模型的训练
- python inference.py   &emsp;   # 模型的推理


#### 13. transformer_reading: 这个项目也是做生成式阅读理解，采用的是标准的transformer结构。
- python train.py   &emsp;  # 模型的训练
- python inference.py  &emsp;  # 模型的推理

## Slot_Filling

## Text_Classification

## Text_Clustering

## Text_Corrector

## Text_Generation

## Text_Ranking

## Text_Similarity

## data_augmentation

## relation_extraction

