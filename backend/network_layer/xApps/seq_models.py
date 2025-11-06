from typing import List

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_ratio: float = 0.5,
        layer_norm: bool = True,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None

        head_hidden = max(1, int(hidden_dim * fc_ratio))
        head: List[nn.Module] = [nn.Linear(hidden_dim, head_hidden), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(head_hidden, output_dim))
        self.fc = nn.Sequential(*head)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        feat = self._features(x)
        if return_features:
            return feat
        return self.fc(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_features=True)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        channels = [input_dim] + [hidden_dim] * num_layers
        layers: List[nn.Module] = []
        dilation = 1
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation,
            )
            layers.append(nn.utils.weight_norm(conv))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # -> (batch, input_dim, seq_len)
        y = self.net(x)
        return y[:, :, -1]

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        feat = self._features(x)
        if return_features:
            return feat
        return self.output(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_features=True)


class Seq2SeqAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_ratio: float = 0.5,
        layer_norm: bool = True,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        attn_input_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(attn_input_dim) if layer_norm else None
        head_hidden = max(1, int(attn_input_dim * fc_ratio))
        head: List[nn.Module] = [nn.Linear(attn_input_dim, head_hidden), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(head_hidden, output_dim))
        self.fc = nn.Sequential(*head)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, (h_n, _) = self.encoder(x)
        last_hidden = h_n[-1]  # (batch, hidden)
        scores = torch.bmm(encoder_outputs, last_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        combined = torch.cat([context, last_hidden], dim=1)
        if self.norm is not None:
            combined = self.norm(combined)
        return combined

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        feat = self._features(x)
        if return_features:
            return feat
        return self.fc(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_features=True)
