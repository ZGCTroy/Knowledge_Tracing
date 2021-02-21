import torch.nn as nn


class BertLanguageModel(nn.Module):
    def __init__(self, embedding_layer, bert_encoder, nsp_decoder, mlm_decoder, task_decoder=None):
        super(BertLanguageModel, self).__init__()
        self.model_name = 'Bert'

        self.embedding_layer = embedding_layer

        self.bert_encoder = bert_encoder

        self.nsp_decoder = nsp_decoder

        self.mlm_decoder = mlm_decoder

        self.task_decoder = task_decoder

    def forward(self, src, segment_info=None, src_mask=None):
        # embedding the indexed sequence to sequence of vectors
        src = self.embedding_layer(inputs=src, segment_info=segment_info)

        encoder_output, attention_weights = self.bert_encoder(src=src, src_mask=src_mask)

        nsp_output = self.nsp_decoder(encoder_output[0])

        mlm_outputs = {}
        for name in self.embedding_layer.input_name:
            mlm_output = self.mlm_decoder[name](encoder_output)
            mlm_output = mlm_output.permute(1, 0, 2)
            mlm_outputs[name] = mlm_output

        task_output = self.task_decoder(encoder_output[0])

        return encoder_output.permute(1, 0, 2), nsp_output, mlm_outputs, task_output, attention_weights


class MlmDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim=2, dropout=0.2):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        return self.decoder(input)


class NspDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        return self.decoder(input)


class TaskDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.decoder(input)
