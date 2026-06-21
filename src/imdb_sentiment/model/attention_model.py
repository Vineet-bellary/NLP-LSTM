import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=128, padding_idx=0
        )
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )
        self.attention = nn.Linear(in_features=128, out_features=1)
        self.fc = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, (_unused_hidden, _unused_cell) = self.lstm(embedded)
        scores = self.attention(output)
        weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(weights * output, dim=1)
        context = self.dropout(context_vector)
        logits = self.fc(context)

        return logits
