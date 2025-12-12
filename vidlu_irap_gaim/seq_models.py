from __future__ import annotations

import torch
from torch import nn


class InputEncoder(nn.Module):
    """
    Base class for input encoders.
    Each encoder processes a specific input type (e.g., labels, logits, feats).
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityEncoder(InputEncoder):
    def __init__(self, input_dim: int):
        super().__init__(output_dim=input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LabelEmbeddingEncoder(InputEncoder):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(output_dim=embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, 1) or (B, S) -> (B, S, E)
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return self.embedding(x.long())


class GeneralLSTMModel(nn.Module):
    """
    A modular LSTM-based sequential enhancement model.
    Matches the architecture of libs/irap_gaim-main/seq_enh_model.py:GeneralLSTMModel

    Key features:
    - Modular input encoders (label embeddings, feats, etc.)
    - LSTM processing of concatenated inputs
    - Aggregation of global context (final hidden state) and local context (middle output)
    """

    def __init__(
        self,
        n_classes: int,
        input_encoders: dict[str, InputEncoder],
        hidden_dim: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        middle_index: int | None = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.input_encoders = nn.ModuleDict(input_encoders)

        # Calculate total input dimension to LSTM
        self.total_input_dim = sum(enc.output_dim for enc in self.input_encoders.values())

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        self.middle_index = middle_index

        self.lstm = nn.LSTM(
            input_size=self.total_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # FC input dim = (num_directions * (num_layers + 1) * hidden_dim)
        # Reason:
        # - Global context: Flattened hidden states of all layers & directions from the LAST time step.
        #   PyTorch LSTM returns `hn` of shape (num_layers * num_directions, B, hidden_dim).
        #   We flatten this to (B, num_layers * num_directions * hidden_dim).
        # - Local context: Output at the middle time step. Shape (B, num_directions * hidden_dim).
        # Total = (num_layers * num_directions + num_directions) * hidden_dim
        #       = num_directions * (num_layers + 1) * hidden_dim

        fc_input_dim = self.n_directions * (self.n_layers + 1) * self.hidden_dim
        self.fc = nn.Linear(fc_input_dim, n_classes)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: Dictionary mapping input type names (matching input_encoders keys) to tensors.
                    Tensors should be (B, S, ...)
        """
        # 1. Encode and concatenate inputs
        encoded_inputs = []
        # Ensure consistent order based on keys
        # Note: In Python 3.7+, dict order is insertion order, but sorting keys is safer if dynamic
        sorted_keys = sorted(self.input_encoders.keys())

        # Check that we have a tensor for every encoder
        # (Or we could allow missing inputs if we want optionality, but strict for now)
        for key in sorted_keys:
            if key not in inputs:
                raise ValueError(f"Missing input for encoder: {key}")
            encoded = self.input_encoders[key](inputs[key])
            encoded_inputs.append(encoded)

        # Concatenate along feature dimension (dim 2)
        # Shape: (B, S, Total_Input_Dim)
        concated_sequence = torch.cat(encoded_inputs, dim=2)

        B, S, _ = concated_sequence.shape

        # 2. LSTM Forward
        # output: (B, S, D * H_out)
        # hn: (D * num_layers, B, H_out)
        output, (hn, _) = self.lstm(concated_sequence)

        # 3. Aggregate Global Context (Hidden States)
        # Transpose hn to (B, D * num_layers, H_out) then flatten to (B, -1)
        # Original code: hn = hn.transpose(0, 1).contiguous().view(B, -1)
        hn = hn.transpose(0, 1).contiguous().view(B, -1)

        # 4. Aggregate Local Context (Middle Output)
        if self.middle_index is not None:
            mid_idx = self.middle_index
        else:
            mid_idx = S // 2

        middle_out = output[:, mid_idx, :]

        # 5. Concatenate and Predict
        final_features = torch.cat([hn, middle_out], dim=1)
        logits = self.fc(final_features)

        return logits
