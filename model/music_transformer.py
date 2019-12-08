import torch
import torch.nn as nn

from utilities.constants import *
from utilities.tensors import create_full_tensor
from .positional_encoding import PositionalEncoding



# MusicTransformer
class MusicTransformer(nn.Module):
    def __init__(self, args):
        super(MusicTransformer, self).__init__()

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
        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff
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
        return y[:,:-1,:]

    # generate
    def generate(self, primer=None):
        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", self.max_seq)

        gen_seq = create_full_tensor((1,self.max_seq), TOKEN_END, TORCH_LABEL_TYPE)

        if(primer is not None):
            num_primer = len(primer)
            gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(TORCH_DEVICE)
        else:
            gen_seq[..., 0] = TOKEN_START
            num_primer = 1

        cur_i = num_primer
        while(cur_i < self.max_seq):
            y           = self.forward(gen_seq)
            next_token  = torch.argmax(y[0, cur_i, :])

            gen_seq[0, cur_i] = next_token

            # Let the transformer decide to end if it wants to
            if(next_token == TOKEN_END):
                print("Model called end of sequence at:", cur_i, "/", self.max_seq)
                break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", self.max_seq)

        # No need to keep the start and end tokens
        return gen_seq[:, 1:cur_i]
