import math

import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Bert(nn.Module):
    def __init__(self, bert_encoder, nsp_decoder, mlm_decoder):
        super(Bert, self).__init__()
        self.bert_encoder = bert_encoder

        self.nsp_decoder = nsp_decoder

        self.mlm_decoder = mlm_decoder

    def forward(self, src, src_mask):
        embedding_vector = self.bert_encoder(src, src_mask)

        nsp_output = self.nsp_decoder(embedding_vector)

        mlm_decoder = self.mlm_decoder(embedding_vector)

        return embedding_vector, nsp_output, mlm_decoder


class MlmDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input):
        return self.decoder(input)


class NspDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input):
        return self.decoder(input)


class BertEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, num_head, hidden_dim, num_encoder_layers, dropout=0.5,
                 max_seq_len=20):
        super(Bert, self).__init__()
        self.model_type = 'MyBertModel'

        # embedding layer : input vector --> embedding vector
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # embedding vector + position vector  --> encoder input
        self.pos_encoder = PositionalEmbedding(embedding_dim, dropout=dropout, max_len=max_seq_len)

        # Transformer Trm : encoder input + mask vector --> encoder output
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_head,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_encoder_layers
        )

        # Output Layer: encoder output --> network output
        self.output_dim = output_dim
        self.decoder = nn.Linear(embedding_dim, output_dim)

        # init weights
        self.init_weights()

    # def generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def init_weights(self):
        init_range = 0.1
        self.embedding_layer.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask=None):
        src = self.embedding_layer(src)
        src = src * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

    def get_embedding_vector(self, src, src_mask):
        src = self.embedding_layer(src)
        src = src * math.sqrt(self.embedding_dim)
        embedding_vector1 = self.pos_encoder(src)

        embedding_vector2 = self.transformer_encoder.layers[0](embedding_vector1, src_mask)
        embedding_vector3 = self.transformer_encoder.layers[1](embedding_vector2, src_mask)

        return embedding_vector1, embedding_vector2, embedding_vector3
