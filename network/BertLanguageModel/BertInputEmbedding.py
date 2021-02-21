import math

import torch
import torch.nn as nn


class BertInputEmbedding(nn.Module):
    """
    BERT Input Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, embedding_dim, input_name=[], input_vocab_size={}, dropout=0.1, use_combine_linear=False):
        """
        :param vocab_size: total vocab size
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        self.tokens = nn.ModuleDict()
        self.num_input = len(input_name)
        self.input_name = input_name
        self.input_vocab_size = input_vocab_size

        for name in input_name:
            self.tokens[name] = TokenEmbedding(vocab_size=input_vocab_size[name]+4, embed_size=embedding_dim, padding_idx=0)

        self.position = PositionalEmbedding(d_model=embedding_dim)
        self.segment = SegmentEmbedding(vocab_size=2 + 1, embed_size=embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim

        self.use_combine_linear = use_combine_linear
        if use_combine_linear:
            self.combine_linear = nn.Linear(in_features=self.num_input+2,out_features=1)

    def forward(self, inputs, segment_info):
        seq_len = inputs[self.input_name[0]].size(1)
        batch_size = inputs[self.input_name[0]].size(0)
        position_embedding = self.position(seq_len)
        position_embedding = torch.repeat_interleave(input=position_embedding, repeats=batch_size,dim=0)
        segment_embedding = self.segment(segment_info)
        stack = [position_embedding, segment_embedding]

        for name in self.input_name:
            stack.append(self.tokens[name](inputs[name]))

        stacked_embedding = torch.stack(stack,dim=3)

        if self.use_combine_linear:
            embedding = self.combine_linear(stacked_embedding).squeeze(-1)
        else:
            embedding = torch.sum(stacked_embedding,dim=3).squeeze(-1)

        return self.dropout(embedding)



class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, padding_idx):
        super().__init__(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, padding_idx):
        super().__init__(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
