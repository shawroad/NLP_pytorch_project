"""

@file  : inference.py

@author: xiaolu

@time  : 2020-04-15

"""
import torch
from torch import nn
import json
from config import Config
from model import Encoder, Decoder, GreedySearchDecoder


def evaluateInput(encoder, decoder, searcher, vocab2id, id2vocab):
    # 随机抽取一条数据
    # 1.数据集整理
    data = json.load(open(Config.train_data_path, 'r'))
    index = 15
    input_data = data['input_data'][index]
    input_len = data['input_len'][index]
    output_data = data['output_data'][index]
    mask_data = data['mask'][index]
    output_len = data['output_len'][index]

    input_batch = torch.LongTensor([input_data])
    lengths = torch.LongTensor([input_len])

    max_len = 62
    tokens, scores = searcher(input_batch, lengths, max_len)

    true_answer = [id2vocab[str(x)] for x in output_data if not (id2vocab[str(x)] == 'EOS' or id2vocab[str(x)] == 'PAD')]
    print('正确答案:', ''.join(true_answer))

    decoder_words = [id2vocab[str(token.item())] for token in tokens]
    decoder_words[:] = [x for x in decoder_words if not (x == 'EOS' or x == 'PAD')]
    print('预测答案:', ''.join(decoder_words))



if __name__ == '__main__':
    # 加载模型
    loadFilename = './save_model/' + 'epoch0_checkpoint_model.tar'
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']

    embedding = nn.Embedding(Config.vocab_size, Config.hidden_size)
    embedding.load_state_dict(embedding_sd)

    encoder = Encoder(embedding)
    attn_model = 'dot'
    decoder = Decoder(attn_model, embedding,)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder = encoder.to(Config.device)
    decoder = decoder.to(Config.device)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    vocab2id = json.load(open('./data/vocab2id.json', 'r'))
    id2vocab = json.load(open('./data/id2vocab.json', 'r'))
    print(id2vocab)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, vocab2id, id2vocab)