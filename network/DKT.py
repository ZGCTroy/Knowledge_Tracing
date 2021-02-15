import torch
import torch.nn as nn

class DKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.model_name = 'DKT'
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.encoder = nn.Sequential(
            self.embedding_layer,
            nn.Dropout(dropout)
        )

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, input, target_id):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        # TODO 1: Encoder
        embedded_input = self.encoder(input)
        LSTM_input = embedded_input

        # TODO 2: LSTM
        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)
        ht, hT = self.LSTM(LSTM_input, (hidden[0].detach(), hidden[1].detach()))
        # hT = ht[:, -1, :]  # the last timestamp

        # TODO 3: Decoder
        decoder_input = ht
        output = self.decoder(decoder_input)

        # TODO 4: Query skill state of skill id
        # 1 : target_id 即 skill id, 取出output 中 第skill_id 的值
        # output = torch.gather(output, -1, target_id)

        output = torch.gather(output, dim=2, index = target_id.unsqueeze(dim=2)).squeeze(-1)

        # 2 : weighted
        # query_embedded = self._encoder(target_id).view(batch_size,-1)
        # output = output * query_embedded
        # output = torch.sum(output,dim=1,keepdim=True)

        return output
