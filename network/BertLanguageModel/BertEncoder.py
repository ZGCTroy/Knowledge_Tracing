import torch.nn as nn

from .BertInputEmbedding import BertInputEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class BertEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=768, num_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param embeddingg_dim: BERT model embedding size
        :param hidden_dim: BERT model hidden size
        :param num_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.attn_heads = attn_heads

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding_layer = BertInputEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attn_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )

    def forward(self, src, segment_info, src_mask):

        # embedding the indexed sequence to sequence of vectors
        src = self.embedding_layer(src, segment_info)

        src = self.transformer_encoder(src.permute(1,0,2), src_key_padding_mask = src_mask)

        return src