"""

@file  : inference.py

@author: xiaolu

@time  : 2020-04-01

"""
import torch
from data_helper import loadPrepareData, trimRareWords
from config import Config
from seq2seq import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from data_helper import normalizeString, indexesFromSentence
from torch import nn


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=Config.MAX_LENGTH):
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # print(input_batch.size())   # torch.Size([7, 1]) 不进行padding seq_len x batch_size

    # Use appropriate device
    input_batch = input_batch.to(Config.device)
    lengths = lengths.to(Config.device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # print(tokens)  # tensor([ 25, 200,   2,   4,   2,   2,   2,   2,   2,   2])
    # print(scores)  # tensor([0.1747, 0.2254, 0.0871, 0.1472, 0.1718, 0.2144, 0.2478, 0.3085, 0.4856, 0.9942])

    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    while(1):
        try:
            # Get input sentence
            # input_sentence = input('> ')
            input_sentence = 'what you name how are you'
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # print(input_sentence)

            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


if __name__ == '__main__':
    data_path = './data/chatbot.txt'
    voc, pairs = loadPrepareData(data_path)

    # 把含有低频词的句子扔掉
    MIN_COUNT = Config.MIN_COUNT
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    loadFilename = './save_model/' + '4000_checkpoint_model.tar'

    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    embedding = nn.Embedding(voc.num_words, Config.hidden_size)
    embedding.load_state_dict(embedding_sd)

    encoder = EncoderRNN(Config.hidden_size, embedding, Config.encoder_n_layers, Config.dropout)
    decoder = LuongAttnDecoderRNN(Config.attn_model, embedding, Config.hidden_size, voc.num_words, Config.decoder_n_layers, Config.dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder = encoder.to(Config.device)
    decoder = decoder.to(Config.device)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, voc)