import torch
import torch.nn as nn
import random

from torch.nn.modules.normalization import LayerNorm
from utilities.constants import *
from utilities.tensors import create_full_tensor
from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


# MusicTransformer
class MusicTransformer(nn.Module):
    def __init__(self, args):
        super(MusicTransformer, self).__init__()

        self.dummy      = DummyDecoder()

        self.nlayers    = args.n_layers
        self.nhead      = args.num_heads
        self.d_model    = args.d_model
        self.d_ff       = args.dim_feedforward
        # self.ff_activ   = args.feedforward_activation
        self.dropout    = args.dropout
        self.max_seq    = args.max_sequence
        self.rpr        = args.rpr

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(TORCH_DEVICE)
        else:
            mask = None

        x = self.embedding(x)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1,0,2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # generate
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = create_full_tensor((1,target_seq_length), TOKEN_PAD, TORCH_LABEL_TYPE)

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(TORCH_DEVICE)


        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            y           = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END]
            token_probs  = y[:, cur_i-1, :]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        return memory
