import torch
import torch.nn as nn
import random

from utilities.constants import *
from utilities.tensors import create_full_tensor
from .positional_encoding import PositionalEncoding



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

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # To make a decoder-only transformer we need to use masked encoder layers
        # Dummy decoder to essentially just return the encoder output
        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_decoder=self.dummy
        )

        # Input sequence mask
        self.mask = self.transformer.generate_square_subsequent_mask(self.max_seq).to(TORCH_DEVICE)

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x):
        x = self.embedding(x)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1,0,2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=self.mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # generate
    def generate(self, primer=None, target_seq_length=1024):
        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = create_full_tensor((1,self.max_seq), TOKEN_END, TORCH_LABEL_TYPE)

        if(primer is not None):
            num_primer = len(primer)
            gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(TORCH_DEVICE)
        else:
            gen_seq[..., 0] = TOKEN_START
            num_primer = 1


        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            y           = self.softmax(self.forward(gen_seq))
            token_probs  = y[0, cur_i-1, :]

            # probability_dist = random.randint(0,1)
            probability_dist=1

            if(probability_dist == 1):
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)

                # Must persist for 5 times at TOKEN_END to end early
                persist = 0
                next_token = -1
                while(persist < 5):
                    next_token = distrib.sample()
                    if(next_token == TOKEN_END):
                        persist += 1
                        print("persist:", persist)
                    else:
                        break
            else:
                next_token = torch.argmax(token_probs, dim=-1)

            # print("next token:",next_token)

            # print(next_token)
            gen_seq[0, cur_i] = next_token


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
