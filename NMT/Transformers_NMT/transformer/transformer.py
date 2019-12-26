"""

@file  : transformer.py

@author: xiaolu

@time  : 2019-12-25

"""
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    '''
    编码＋解码
    '''
    def __init__(self, encoder=None, decoder=None):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input:
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        return pred, gold

    def recognize(self, input, input_length, char_list):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, enc_slf_attn_list = self.encoder(padded_input=input.unsqueeze(0), input_lengths=input_length,
                                                          return_attns=True)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0], char_list)
        return nbest_hyps
