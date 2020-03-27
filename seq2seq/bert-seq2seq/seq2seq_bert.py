"""

@file  : seq2seq_bert.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch
import torch.nn as nn
from bert_model import BertModel, BertConfig, BertLMPredictionHead
from tokenizer import Tokenizer, load_bert_vocab
from config import Config


class Seq2SeqModel(nn.Module):
    """
    模型
    """
    def __init__(self, config: BertConfig):
        super(Seq2SeqModel, self).__init__()
        # 获取配置信息
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size

        # encoder and decoder
        self.bert = BertModel(config)
        self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

        # 加载字典和分词器
        self.word2ix = load_bert_vocab()
        self.tokenizer = Tokenizer(self.word2ix)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()  ## 通过mask 取消 pad 和句子a部分预测的影响

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None, device="cpu"):
        '''
        :param input_tensor: 传入输入
        :param token_type_id: 句子标志
        :param position_enc: 位置编码
        :param labels: 解码的句子
        :param device:
        :return:
        '''
        input_shape = input_tensor.size()

        seq_len = input_shape[1]
        # 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=device)
        a_mask = ones.tril()  # 下三角矩阵
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        # print(a_mask.size())   # torch.Size([2, 1, 44, 44])

        enc_layers, _ = self.bert(input_tensor,
                                  position_ids=position_enc,
                                  token_type_ids=token_type_id,
                                  attention_mask=a_mask,
                                  output_all_encoded_layers=True)
        # print(_.size())  # torch.Size([2, 768])   (batch_size, hidden_size)

        squence_out = enc_layers[-1]  # 取出来最后一层输出
        # print(squence_out.size())    # torch.Size([2, 31, 768])

        predictions = self.decoder(squence_out)
        # print(labels.size())   # torch.Size([2, 30])
        # print(predictions.size())   # torch.Size([2, 31, 21128])

        if labels is not None:
            # 计算loss
            # 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            predictions = predictions[:, :-1].contiguous()
            # print(predictions.size())  # torch.Size([2, 30, 21128])

            target_mask = token_type_id[:, 1:].contiguous()
            # print(target_mask)
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=50, beam_size=1, device="cpu"):
        # 对一个句子生成相应的结果
        # 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        self.out_max_length = out_max_length
        input_max_length = Config.max_length - out_max_length
        # print(text)
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=device).view(1, -1)
        out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=device)
        # 解码 得到相应输出
        return self.tokenizer.decode(out_puts_ids)

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        sep_id = word2ix["[SEP]"]
        # 用来保存输出序列
        output_ids = [[]]
        # 用来保存累计得分
        output_scores = torch.zeros(token_ids.shape[0], device=device)
        for step in range(self.out_max_length):

            scores = self.forward(token_ids, token_type_ids, device=device)
            # print(scores.shape)
            if step == 0:
                # 重复beam-size次 输入ids
                token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
            ## 计算log 分值 (beam_size, vocab_size)
            logit_score = torch.log_softmax(scores, dim=-1)[:, -1]
            logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
            ## 取topk的时候我们是展平了然后再去调用topk函数
            # 展平
            logit_score = logit_score.view(-1)
            hype_score, hype_pos = torch.topk(logit_score, beam_size)
            indice1 = hype_pos / scores.shape[-1]  # 行索引
            indice2 = hype_pos % scores.shape[-1]  # 列索引

            # 下面需要更新一下输出了
            new_hype_scores = []
            new_hype_ids = []
            # 为啥有这个[],就是因为要过滤掉结束的序列。
            next_chars = []  # 用来保存新预测出来的一个字符，继续接到输入序列后面，再去预测新字符
            for i_1, i_2, score in zip(indice1, indice2, hype_score):
                i_1 = i_1.item()
                i_2 = i_2.item()
                socre = score.item()

                hype_id = output_ids[i_1] + [i_2]  # 保存所有输出的序列，而不仅仅是新预测的单个字符

                if i_2 == sep_id:
                    # 说明解码到最后了
                    if score == torch.max(hype_score).item():
                        # 说明找到得分最大的那个序列了 直接返回即可
                        return hype_id[: -1]
                    else:
                        # 完成一个解码了，但这个解码得分并不是最高，因此的话需要跳过这个序列
                        beam_size -= 1
                else:
                    new_hype_ids.append(hype_id)
                    new_hype_scores.append(score)
                    next_chars.append(i_2)  # 收集一下，需要连接到当前的输入序列之后

            output_ids = new_hype_ids

            output_scores = torch.tensor(new_hype_scores, dtype=torch.float32, device=device)
            # 现在需要重新构造输入数据了，用上一次输入连接上这次新输出的字符，再输入bert中预测新字符
            token_ids = token_ids[:len(output_ids)].contiguous()  # 截取，因为要过滤掉已经完成预测的序列
            token_type_ids = token_type_ids[: len(output_ids)].contiguous()

            next_chars = torch.tensor(next_chars, dtype=torch.long, device=device).view(-1, 1)
            next_token_type_ids = torch.ones_like(next_chars, device=device)
            # 连接
            token_ids = torch.cat((token_ids, next_chars), dim=1)
            token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)
            if beam_size < 1:
                break

        # 如果达到最大长度的话 直接把得分最高的输出序列返回把
        return output_ids[output_scores.argmax().item()]






