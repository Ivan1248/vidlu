"""
Legacy sequential enhancement models.

This module is intentionally compatible with checkpoints produced by the original
`libs/irap_gaim-main` codebase.

Only the minimal subset needed for inference is included here.

Used by:
- `vidlu_irap_gaim.tools.inference_visualization.run_sequential_legacy()` for loading
  legacy per-attribute LSTM smoothing checkpoints in sequential_legacy mode.
"""

import torch
from torch import nn


class LegacyGeneralLSTMModel(nn.Module):
    """Compatibility port of `libs/irap_gaim-main/seq_enh_model.py:GeneralLSTMModel`.

    Notes:
    - Expects inputs shaped as (B, 2*max_N+1, D_in) for each input type.
    - Internally slices the window to (2*N+1) centered at `max_N`.
    - State dict keys match the legacy implementation (e.g. `label_embedding.weight`, `lstm.*`, `fc.*`).
    """

    def __init__(self, max_N, N, n_classes, input_type_to_metadata, lstm_hyperparams=None):
        super().__init__()
        self.max_N = int(max_N)
        self.N = int(N)
        self._initial_N = int(N)
        self.n_classes = int(n_classes)

        self.input_type_to_metadata = dict(input_type_to_metadata)
        input_type_to_layers = {}

        if "label" in self.input_type_to_metadata:
            layer_dict = self._create_label_layer(self.input_type_to_metadata)
            if "module" in layer_dict:
                self.add_module(*layer_dict["module"])
            input_type_to_layers["label"] = layer_dict

        if "logit" in self.input_type_to_metadata:
            layer_dict = self._create_logit_layer(self.input_type_to_metadata)
            if "module" in layer_dict:
                self.add_module(*layer_dict["module"])
            input_type_to_layers["logit"] = layer_dict

        if "feats" in self.input_type_to_metadata:
            layer_dict = self._create_feat_layer(self.input_type_to_metadata)
            if "module" in layer_dict:
                self.add_module(*layer_dict["module"])
            input_type_to_layers["feats"] = layer_dict

        if "multi_logit" in self.input_type_to_metadata:
            layer_dict = self._create_multi_logit_layer(self.input_type_to_metadata)
            if "module" in layer_dict:
                self.add_module(*layer_dict["module"])
            input_type_to_layers["multi_logit"] = layer_dict

        if "street_level" in self.input_type_to_metadata:
            raise NotImplementedError("legacy 'street_level' input is not supported in vidlu_irap_gaim compat model.")

        self.input_type_to_layers = input_type_to_layers
        self.total_input_dim = sum(m["output_dim"] for m in self.input_type_to_layers.values())
        self.lstm_input_dim = self.total_input_dim

        self.lstm_hyperparams = lstm_hyperparams
        if self.lstm_hyperparams is None:
            self.lstm_hidden_dim = 16
            self.lstm_n_layers = 2
            self.lstm_n_directions = 2
            self.only_before_context = False
        else:
            self.lstm_hidden_dim = int(self.lstm_hyperparams["hidden_dim"])
            self.lstm_n_layers = int(self.lstm_hyperparams["n_layers"])
            self.lstm_n_directions = int(self.lstm_hyperparams["n_directions"])
            self.only_before_context = bool(self.lstm_hyperparams["only_before_context"])

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_n_layers,
            batch_first=True,
            bidirectional=(self.lstm_n_directions == 2),
        )

        fc_input_dim = self.lstm_n_directions * (self.lstm_n_layers + 1) * self.lstm_hidden_dim
        self.fc = nn.Linear(fc_input_dim, self.n_classes)

    def _create_label_layer(self, input_type_to_metadata):
        layer_dict = {}
        layer_dict["output_dim"] = int(input_type_to_metadata["label"]["embedding_dim"])
        layer_dict["module"] = ("label_embedding", nn.Embedding(self.n_classes, layer_dict["output_dim"]))
        return layer_dict

    def _create_logit_layer(self, _input_type_to_metadata):
        layer_dict = {}
        layer_dict["output_dim"] = self.n_classes
        return layer_dict

    def _create_multi_logit_layer(self, input_type_to_metadata):
        layer_dict = {}
        layer_dict["output_dim"] = int(input_type_to_metadata["multi_logit"]["output_dim"])
        return layer_dict

    def _create_feat_layer(self, input_type_to_metadata):
        layer_dict = {}
        layer_dict["output_dim"] = int(input_type_to_metadata["feats"]["feats_dim"])
        return layer_dict

    def set_N(self, new_N):
        self.N = int(new_N)

    def prepare_input(self, input_type_to_batch, mode, device):
        prepared = {}
        for input_type, input_data_batch in input_type_to_batch.items():
            prepared[input_type] = input_data_batch.to(device)
        return prepared

    def _featurize_sequence(self, input_data):
        input_type_to_featurized = {}

        start_idx = self.max_N - self.N
        end_idx = self.max_N + self.N + 1
        if self.only_before_context:
            end_idx = self.max_N + 1

        for input_type, input_data_batch in input_data.items():
            input_data_batch = input_data_batch[:, start_idx:end_idx]
            layer_metadata = self.input_type_to_layers[input_type]
            if "module" in layer_metadata:
                featurized = layer_metadata["module"][1](input_data_batch)
            else:
                featurized = input_data_batch
            input_type_to_featurized[input_type] = featurized

        concated_sequence = torch.cat(list(input_type_to_featurized.values()), dim=2)
        return concated_sequence

    def forward(self, input_data):
        concated_sequence = self._featurize_sequence(input_data)
        B = concated_sequence.size(0)

        output, (hn, _) = self.lstm(concated_sequence)
        hn = hn.transpose(0, 1).contiguous().view(B, -1)
        middle_out = output[:, self.N, :]
        concated = torch.cat([hn, middle_out], dim=1)
        logits = self.fc(concated)
        return logits













