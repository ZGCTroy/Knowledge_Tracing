import torch
import torch.nn as nn

PAD_INDEX = 0


class DKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        self._encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_INDEX)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self._decoder = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self._num_layers, batch_size, self._hidden_dim),
                weight.new_zeros(self._num_layers, batch_size, self._hidden_dim))

    def forward(self, input, target_id, use_pretrained_embedding=False, traditional=True):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)

        embedded_input = input
        embedded_query_question = query_question

        if not use_pretrained_embedding:
            embedded_input = self._encoder(input)
            embedded_query_question = self._encoder(query_question)

        output, _ = self._lstm(embedded_input, (hidden[0].detach(), hidden[1].detach()))

        # hidden_dim --> question_embedding_dim (skill num)
        output = self._decoder(
            output[:, -1, :])  # (batch_size, embedding_dim) ,embedding_dim = skill_num 表示该学生当前的对所有知识的掌握水平

        if traditional:
            output = torch.gather(output, -1, target_id)
        else:
            output = torch.dot(output, embedded_query_question)

        output = torch.nn.Sigmoid(output)

        return output
