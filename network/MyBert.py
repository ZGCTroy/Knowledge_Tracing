import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MyBertModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, num_head, num_hidden, num_encoder_layers, dropout=0.5):
        super(MyBertModel, self).__init__()
        self.model_type = 'MyBertModel'

        # embedding layer : input vector --> embedding vector
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # embedding vector + position vector  --> encoder input
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Transformer Trm : encoder input + mask vector --> encoder output
        encoder_layers = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_head,
            dim_feedforward=num_hidden,
            dropout=dropout
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_encoder_layers
        )

        # Output Layer: encoder output --> network output
        self.output_dim = output_dim
        self.decoder = nn.Linear(embedding_dim, output_dim)

        # init weights
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.embedding_layer.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.embedding_layer(src)
        src = src * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def get_embedding_vector(self, src, src_mask):
        src = self.embedding_layer(src)
        src = src * math.sqrt(self.embedding_dim)
        embedding_vector1 = self.pos_encoder(src)

        embedding_vector2 = self.transformer_encoder.layers[0](embedding_vector1, src_mask)
        embedding_vector3 = self.transformer_encoder.layers[1](embedding_vector2, src_mask)

        return embedding_vector1, embedding_vector2, embedding_vector3


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
