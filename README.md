- [NLP_pytorch_project](#nlp_pytorch_project)
  - [Chatbot](#chatbot)
      - [1. Bert_chatbot: 类似于UniLM的方式](#1-bert_chatbot-类似于unilm的方式)
      - [2. seq2seq_luong: 编码器为两层gru网络，解码器为一层gru网络，在编码器和解码器中间，加入了luong注意力。](#2-seq2seq_luong-编码器为两层gru网络解码器为一层gru网络在编码器和解码器中间加入了luong注意力)
      - [3. transformer_chatbot: 标准的transformer模型](#3-transformer_chatbot-标准的transformer模型)
  - [Distillation](#distillation)
      - [1. DynaBert: 华为的工作，主要是采用剪枝的方式对bert的某些结构进行裁剪。](#1-dynabert-华为的工作主要是采用剪枝的方式对bert的某些结构进行裁剪)
      - [2. rnn_distill_bert: 用一层lstm网络去蒸馏bert模型，只加入了软标签损失。](#2-rnn_distill_bert-用一层lstm网络去蒸馏bert模型只加入了软标签损失)
      - [3. three_layer_self-attention_to_distill_bert: 看名字大概都知道啥意思了，就是写了三层transformer的encoder，然后去蒸馏bert模型](#3-three_layer_self-attention_to_distill_bert-看名字大概都知道啥意思了就是写了三层transformer的encoder然后去蒸馏bert模型)
      - [4. tiny_bert: 华为的工作，tiny_bert的蒸馏方式是除了加入软标签损失，还加入了中间层的均方误差损失](#4-tiny_bert-华为的工作tiny_bert的蒸馏方式是除了加入软标签损失还加入了中间层的均方误差损失)
  - [Embedding](#embedding)
      - [1. skipgram-word2vec: 使用skipgram的方式得到词向量](#1-skipgram-word2vec-使用skipgram的方式得到词向量)
      - [2. bert: 直接训练bert, 从头训练， 也可以使用此代码进行再训练](#2-bert-直接训练bert-从头训练-也可以使用此代码进行再训练)
      - [3. albert: 直接训练albert，从头训练， 也可以使用此代码进行再训练](#3-albert-直接训练albert从头训练-也可以使用此代码进行再训练)
      - [4. NPLM: 传统方法](#4-nplm-传统方法)
  - [NER](#ner)
      - [1. Bert_CRF_Ner: bert模型加条件随机场进行序列标注任务](#1-bert_crf_ner-bert模型加条件随机场进行序列标注任务)
      - [2. Bert_Softmax_Ner: 直接使用bert模型进行序列标注](#2-bert_softmax_ner-直接使用bert模型进行序列标注)
      - [3. BiLSTM_CRF_Ner: 使用双向的lstm网络和crf进行序列标注任务](#3-bilstm_crf_ner-使用双向的lstm网络和crf进行序列标注任务)
  - [NMT](#nmt)
      - [1. GRU_attention: 编码器和解码器都是gru网络，中间加入一个普通的注意力机制(直接的加权和)](#1-gru_attention-编码器和解码器都是gru网络中间加入一个普通的注意力机制直接的加权和)
      - [2. Transformers_NMT: 标准的transformer结构做机器翻译](#2-transformers_nmt-标准的transformer结构做机器翻译)
  - [Pretrain_Model](#pretrain_model)
      - [1. bert-pretrain: bert模型的再训练，首先通过get_train_data.py进行数据预处理，包含15%的词进行mask等操作，然后在进行训练。](#1-bert-pretrain-bert模型的再训练首先通过get_train_datapy进行数据预处理包含15的词进行mask等操作然后在进行训练)
      - [2. wobert-pretrain: wobert的预训练模型由苏神给出，这里的再训练可以加入自己构建的词表，然后修改了bert的分词方式。](#2-wobert-pretrain-wobert的预训练模型由苏神给出这里的再训练可以加入自己构建的词表然后修改了bert的分词方式)
  - [Reading_comprehension](#reading_comprehension)
      - [1. BERT_MRC: 使用bert去做机器阅读理解任务。这里长预料采用直接阶段的方式。](#1-bert_mrc-使用bert去做机器阅读理解任务这里长预料采用直接阶段的方式)
      - [2. BiDAF: 双向注意力流机制的机器阅读理解模型](#2-bidaf-双向注意力流机制的机器阅读理解模型)
      - [3. DocQA: 传统的模型](#3-docqa-传统的模型)
      - [4. Match_LSTM: 传统的模型，单纯的rnn结构。](#4-match_lstm-传统的模型单纯的rnn结构)
      - [5. QANet: 也是一个比较传统的模型，但是这个模型是第一个抛弃rnn结构的mrc模型，这个模型也是第一个将self-attention机制引入到mrc任务中的。](#5-qanet-也是一个比较传统的模型但是这个模型是第一个抛弃rnn结构的mrc模型这个模型也是第一个将self-attention机制引入到mrc任务中的)
      - [6. RNet: 传统的模型](#6-rnet-传统的模型)
      - [7. Recurrence-hotpot-baseline: 首次产生用rnn结构处理多跳推理的问题，在hotpotqa数据集中，除了包含答案的预测，还有支撑事实的预测，以及相关段落的预测。](#7-recurrence-hotpot-baseline-首次产生用rnn结构处理多跳推理的问题在hotpotqa数据集中除了包含答案的预测还有支撑事实的预测以及相关段落的预测)
      - [8. albert_mrc: 使用albert预训练模型去做mrc任务](#8-albert_mrc-使用albert预训练模型去做mrc任务)
      - [9. electra_bert: 使用electra预训练模型去做mrc任务](#9-electra_bert-使用electra预训练模型去做mrc任务)
      - [10. mrc_baseline: 如果做mrc任务，建议首先看这个代码，这里面包含了mrc注意的各种细节，如长文本的处理(滑动窗口)， 答案的排序，对抗训练等等。](#10-mrc_baseline-如果做mrc任务建议首先看这个代码这里面包含了mrc注意的各种细节如长文本的处理滑动窗口-答案的排序对抗训练等等)
      - [11. roberta_mrc: 使用roberta预训练模型去做mrc任务](#11-roberta_mrc-使用roberta预训练模型去做mrc任务)
      - [12. transformer+rnn+attention: 这个项目是做生成式的阅读理解，直接采用的是seq2seq的方式，编码器采用的是transformer的encoder, 解码器采用的gru的结构，中间还加入了一层普通的注意力机制。](#12-transformerrnnattention-这个项目是做生成式的阅读理解直接采用的是seq2seq的方式编码器采用的是transformer的encoder-解码器采用的gru的结构中间还加入了一层普通的注意力机制)
      - [13. transformer_reading: 这个项目也是做生成式阅读理解，采用的是标准的transformer结构。](#13-transformer_reading-这个项目也是做生成式阅读理解采用的是标准的transformer结构)
  - [Slot_Filling](#slot_filling)
      - [1. JointBert: 涉及到意图分类和槽分类。直接用bert对输入进行编码，用“CLS”向量进行意图分类。用每个token的最终编码向量进行槽分类。](#1-jointbert-涉及到意图分类和槽分类直接用bert对输入进行编码用cls向量进行意图分类用每个token的最终编码向量进行槽分类)
  - [Text_Classification](#text_classification)
      - [1. DPCNN: 深层的卷积网络+残差的连接使得此模型比以往的CNN结构都好，并且复杂度也不高。](#1-dpcnn-深层的卷积网络残差的连接使得此模型比以往的cnn结构都好并且复杂度也不高)
      - [2. FastBert: 采用一种自蒸馏的方式加快模型的推理。主要应用在分类任务上。](#2-fastbert-采用一种自蒸馏的方式加快模型的推理主要应用在分类任务上)
      - [3. FastText: 由facebook提出，是一种高效的文本分类模型。](#3-fasttext-由facebook提出是一种高效的文本分类模型)
      - [4. XLNet: 1）通过最大化所有可能的因式分解顺序的对数似然，学习双向语境信息；2）用自回归本身的特点克服 BERT 的缺点。此外，XLNet 还融合了当前最优自回归模型 Transformer-XL 的思路。](#4-xlnet-1通过最大化所有可能的因式分解顺序的对数似然学习双向语境信息2用自回归本身的特点克服-bert-的缺点此外xlnet-还融合了当前最优自回归模型-transformer-xl-的思路)
      - [5. all_layer_out_concat: 从名字上就可以看出，本项目是将文本经过bert-style模型进行编码，然后将每层的cls向量拿出来，进行一个注意力计算，然后进行分类。](#5-all_layer_out_concat-从名字上就可以看出本项目是将文本经过bert-style模型进行编码然后将每层的cls向量拿出来进行一个注意力计算然后进行分类)
      - [6. bert+bceloss+average_checkpoint: 该项目将分类任务的损失函数改成了BCELoss，另外，加入的权重平均(将多个checkpoint进行平均)](#6-bertbcelossaverage_checkpoint-该项目将分类任务的损失函数改成了bceloss另外加入的权重平均将多个checkpoint进行平均)
      - [7. capsule_text_classification: GRU+Capsule进行文本分类](#7-capsule_text_classification-grucapsule进行文本分类)
      - [8. longformer_classification](#8-longformer_classification)
      - [9. multi_label_classify_bert](#9-multi_label_classify_bert)
      - [10. roberta_classification](#10-roberta_classification)
      - [11. transformer_xl](#11-transformer_xl)
      - [12. wobert+focal_loss](#12-wobertfocal_loss)
      - [13. TextCNN](#13-textcnn)
      - [14. BILSTM+Attention](#14-bilstmattention)
  - [Text_Clustering](#text_clustering)
  - [Text_Corrector](#text_corrector)
  - [Text_Generation](#text_generation)
  - [Text_Ranking](#text_ranking)
  - [Text_Similarity](#text_similarity)
  - [data_augmentation](#data_augmentation)
  - [relation_extraction](#relation_extraction)# NLP_pytorch_project

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
#### 1. JointBert: 涉及到意图分类和槽分类。直接用bert对输入进行编码，用“CLS”向量进行意图分类。用每个token的最终编码向量进行槽分类。
- python train.py  &emsp; # 对模型的训练

## Text_Classification
#### 1. DPCNN: 深层的卷积网络+残差的连接使得此模型比以往的CNN结构都好，并且复杂度也不高。
- python get_data_to_examples.py  &emsp; # 预处理数据
- python examples_to_features.py  &emsp;  # 将对应的example转为feature
- python train.py   &emsp;   # 模型的训练

#### 2. FastBert: 采用一种自蒸馏的方式加快模型的推理。主要应用在分类任务上。
- sh train_stage0.sh   &emsp;  # 训练老师模型 bert
- sh train_stage1.sh  &emsp;  # 自蒸馏
- sh infer_sigle.sh   &emsp;  # 自适应的推理单条样本

#### 3. FastText: 由facebook提出，是一种高效的文本分类模型。
- python step1_get_data_to_examples.py  &emsp; # 获取数据
- python step2_examples_to_features.py   &emsp;  # 将文本数据转为id序列
- python train.py   &emsp; # 模型训练

#### 4. XLNet: 1）通过最大化所有可能的因式分解顺序的对数似然，学习双向语境信息；2）用自回归本身的特点克服 BERT 的缺点。此外，XLNet 还融合了当前最优自回归模型 Transformer-XL 的思路。
- python train.py   &emsp;  # 模型训练


#### 5. all_layer_out_concat: 从名字上就可以看出，本项目是将文本经过bert-style模型进行编码，然后将每层的cls向量拿出来，进行一个注意力计算，然后进行分类。
- python train.py   &emsp;  # 模型训练
- Python inference.py  &emsp; # 模型推理

#### 6. bert+bceloss+average_checkpoint: 该项目将分类任务的损失函数改成了BCELoss，另外，加入的权重平均(将多个checkpoint进行平均)
- python run_classify.py   &emsp; # 模型训练
- python run_average_checkpoints.py   &emsp;  # 权重平均

#### 7. capsule_text_classification: GRU+Capsule进行文本分类
- python train.py   &emsp;  # 模型训练

#### 8. longformer_classification

#### 9. multi_label_classify_bert


#### 10. roberta_classification


#### 11. transformer_xl

#### 12. wobert+focal_loss


#### 13. TextCNN

#### 14. BILSTM+Attention



## Text_Clustering

## Text_Corrector

## Text_Generation

## Text_Ranking

## Text_Similarity

## data_augmentation

## relation_extraction

