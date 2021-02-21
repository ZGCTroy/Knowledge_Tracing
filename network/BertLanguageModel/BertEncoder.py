import torch.nn as nn
from typing import Optional, Any
from torch import Tensor
from .BertInputEmbedding import BertInputEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MyTransformerEncoder(nn.TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        attention_weights = []
        for mod in self.layers:
            output, attention_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights.append(attention_weight)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_weights

class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = output[0]
        attention_weight = output[1]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention_weight

class BertEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, embedding_dim=768, hidden_dim=768, num_layers=12, attn_heads=12, dropout=0.1):
        """
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

        # multi-layers transformer blocks, deep network
        self.transformer_encoder = MyTransformerEncoder(
            encoder_layer=MyTransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attn_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='gelu'
            ),
            num_layers=num_layers
        )

    def forward(self, src, src_mask):

        src, attention_weights = self.transformer_encoder(src.permute(1,0,2), src_key_padding_mask = src_mask)

        return src, attention_weights