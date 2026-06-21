import torch
import torch.nn as nn


class LSTMModel(nn.Module):
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
        )
        self.fc = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        _unused_output, (hidden, _unused_cell) = self.lstm(embedded)
        last_hidden_state = hidden[-1]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.fc(last_hidden_state)

        # print(f"Embedded shape: {embedded.shape}")
        # print(f"LSTM output shape: {output.shape}")
        # print(f"LSTM hidden shape: {hidden.shape}")
        # print(f"LSTM cell shape: {cell.shape}")
        # print(f"Logits shape: {logits.shape}")

        return logits
