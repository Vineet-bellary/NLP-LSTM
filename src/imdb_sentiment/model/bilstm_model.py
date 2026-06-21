import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=128, padding_idx=0
        )
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        _unused_output, (hidden, _unused_cell) = self.lstm(embedded)
        sentence_rep = torch.cat((hidden[-2], hidden[-1]), dim=1)
        sentence_rep = self.dropout(sentence_rep)
        logits = self.fc(sentence_rep)

        return logits
