"""Core stock prediction backbone used by the responsibility model.

This module is a path-independent extraction of the frozen backbone used by
the retained experiment.  Layer names and construction order are intentionally
kept unchanged so existing checkpoints remain loadable.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList(
            [Dropout(p=dropout) for _ in range(nhead)]
        )
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]
            attention = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / self.temperature,
                dim=-1,
            )
            attention = self.attn_dropout[i](attention)
            att_output.append(torch.matmul(attention, vh).transpose(0, 1))
        att_output_tensor = torch.concat(att_output, dim=-1)
        xt = self.norm2(x + att_output_tensor)
        return xt + self.ffn(xt)


class TAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = []
        if dropout > 0:
            self.attn_dropout = nn.ModuleList(
                [Dropout(p=dropout) for _ in range(nhead)]
            )
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)
        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]
            attention = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)), dim=-1
            )
            if self.attn_dropout:
                attention = self.attn_dropout[i](attention)
            att_output.append(torch.matmul(attention, vh))
        att_output_tensor = torch.concat(att_output, dim=-1)
        xt = self.norm2(x + att_output_tensor)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    def __init__(self, d_input: int, d_output: int, beta: float = 1.0) -> None:
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input: Tensor) -> Tensor:
        logits = self.trans(gate_input)
        return self.d_output * torch.softmax(logits / self.t, dim=-1)


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)


class MASTER(nn.Module):
    """Frozen prediction backbone used by the responsibility modules."""

    def __init__(
        self,
        d_feat: int,
        d_model: int,
        t_nhead: int,
        s_nhead: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        gate_input_start_index: int,
        gate_input_end_index: int,
        beta: float,
    ) -> None:
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        src = x[:, :, : self.gate_input_start_index]
        market = x[
            :, -1, self.gate_input_start_index : self.gate_input_end_index
        ]
        src = src * self.feature_gate(market).unsqueeze(1)
        return self.layers(src).squeeze(-1)


__all__ = [
    "PositionalEncoding",
    "SAttention",
    "TAttention",
    "Gate",
    "TemporalAttention",
    "MASTER",
]
