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

    def __init__(self, vocab_size, embedding_dim, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embedding_dim, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embedding_dim)
        self.segment = SegmentEmbedding(vocab_size=3, embed_size=embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim

    def forward(self, sequence, segment_label):
        x = self.token(sequence)
        x = x + self.position(sequence)
        x = x + self.segment(segment_label)
        return self.dropout(x)



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

    def forward(self, x):
        return self.pe[:, :x.size(1)]



class SegmentEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, padding_idx):
        super().__init__(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
