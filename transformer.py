import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, device: str = "cpu"):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ScaledDotAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dk: int,
        dv: int,
        apply_Wo: bool = True,
        device: str = "cpu",
    ):
        super(ScaledDotAttention, self).__init__()

        self.input_projection = AttentionProjection(model_dim, dk, dv, device)
        self.dk = torch.tensor(dk, dtype=torch.float32, device=device)
        if apply_Wo:
            self.Wo = nn.Linear(dv, model_dim, device=device)
        else:
            self.Wo = nn.Identity()

        self.device = device

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        apply_mask: bool = False,
    ) -> torch.Tensor:
        B = keys.shape[0]
        q, k, v = input_projection(query, keys, values)
        
        scores = torch.bmm(query, keys.permute(0, 2, 1)) / torch.sqrt(self.dk)
        
        if apply_mask:
            T = query.shape[1]
            mask = torch.triu(
                torch.ones(B, T, T, dtype=torch.bool, device=self.device), 1
            )
            scores[mask] = float("-inf")
        
        weights = nn.functional.softmax(scores, -1)
        context = torch.bmm(weights, values)
        context = self.Wo(context)
        
        return context


class MultiHeadAttention(nn.Module):
    def __init__(
        self, model_dim: int, dk: int, dv: int, heads: int, device: str = "cpu"
    ):
        super(MultiHeadAttention, self).__init__()
        assert dk % heads == 0
        assert dv % heads == 0

        self.att_list = nn.ModuleList(
            [
                ScaledDotAttention(
                    model_dim,
                    int(dk / heads),
                    int(dv / heads),
                    apply_Wo=False,
                    device=device,
                )
                for _ in range(heads)
            ]
        )
        self.Wo = nn.Linear(dv, model_dim, device=device)

    def forward(
        self,
        query: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        apply_mask: bool = False,
    ) -> torch.Tensor:
        context = [
            att(q, k, v, apply_mask)
            for att, q, k, v in zip(self.att_list, query, keys, values)
        ]
        context = self.Wo(torch.cat(context, dim=-1))
        return context


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, pe_dim: int, device: str = "cpu"):
        """
        Pre-computed for speed, figure out actual x_length during inference.

        max_seq_len: max number of words in a sequence
        dim: position encoding dim which equals last dimension of input
        """
        super(PositionEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim = pe_dim

        assert pe_dim % 2 == 0

        d = int(pe_dim / 2)
        self.pe = torch.empty(max_seq_len, pe_dim, dtype=torch.float32, device=device)
        for k in range(max_seq_len):
            g = k / (10000 ** (2 * torch.arange(d) / pe_dim))
            self.pe[k, :d] = torch.sin(g)
            self.pe[k, d:] = torch.cos(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect `x` to be tensor of shape [T,F] or [B,T,F].
        """
        if x.dim() == 2:
            k_dim = 0
            i_dim = 1
        elif x.dim() == 3:
            k_dim = 1
            i_dim = 2

        assert x.shape[k_dim] <= self.max_seq_len
        assert x.shape[i_dim] == self.dim
        xk_dim = x.shape[k_dim]

        x = x + self.pe[:xk_dim, :]
        return x


class InputLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        max_seq_len: int,
        dropout_p: float = 0.2,
        device: str = "cpu",
    ):
        super(InputLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_dim, model_dim, device=device)
        self.position_encoding = PositionEncoding(max_seq_len, model_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        return x


class AttentionProjection(nn.Module):
    def __init__(self, model_dim: int, dk: int, dv: int, device: str = "cpu"):
        super(AttentionProjection, self).__init__()
        self.Wq = nn.Linear(model_dim, dk, device=device)
        self.Wk = nn.Linear(model_dim, dk, device=device)
        self.Wv = nn.Linear(model_dim, dv, device=device)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.Wq(x)
        K = self.Wk(y) 
        V = self.Wv(z)
        
        return q, K, V


class EncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        keys_dim: int,
        values_dim: int,
        attention_heads: int = 1,
        device: str = "cpu",
    ) -> None:
        super(Encoder, self).__init__()

        self.selfatt = MultiHeadAttention(
            model_dim, keys_dim, values_dim, attention_heads, device
        )
        self.selfatt_norm = nn.LayerNorm(model_dim, device=device)
        self.ffn = FFN(model_dim, device=device)
        self.ffn_norm = nn.LayerNorm(model_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.selfatt(x, x, x, False)
        x = self.selfatt_norm(x + c)
        y = self.ffn(x)
        x = self.ffn_norm(x + y)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        keys_dim: int,
        values_dim: int,
        attention_heads: int = 1,
        device: str = "cpu",
    ):
        super(Decoder, self).__init__()
        self.dk = keys_dim
        self.dv = values_dim

        self.selfatt = MultiHeadAttention(
            model_dim, self.dk, self.dv, attention_heads, device
        )
        self.selfatt_norm = nn.LayerNorm(model_dim, device=device)

        self.edatt = MultiHeadAttention(
            model_dim, self.dk, self.dv, attention_heads, device
        )
        self.edatt_norm = nn.LayerNorm(model_dim, device=device)

        self.ffn = FFN(model_dim, device=device)
        self.ffn_norm = nn.LayerNorm(model_dim, device=device)

    def forward(
        self, inputs: torch.Tensor, encodings: torch.Tensor, apply_mask: bool = False
    ) -> torch.Tensor:
        x = self.selfatt(inputs, inputs, inputs, apply_mask)
        x = self.selfatt_norm(x + inputs)
        y = self.edatt(x, encodings, encodings, False)
        y = self.edatt_norm(y + x)
        z = self.ffn(y)
        z = self.ffn_norm(z + y)

        return z


class Transformer(nn.Module):
    def __init__(self, args: dict):
        super(Transformer, self).__init__()

        self.args = args
        self.device = args["device"]
        self.epoch = 0

        # Encoder
        self.encoder_input_layer = InputLayer(
            args["input_vocab_size"],
            args["model_dim"],
            args["max_seq_len"],
            args["input_dropout"],
            self.device,
        )
        self.encoder_stack = nn.Sequential(
            *[
                EncoderLayer(
                    args["model_dim"],
                    args["keys_dim"],
                    args["values_dim"],
                    args["heads"],
                    self.device,
                )
                for _ in range(args["encoder_stack_size"])
            ]
        )

        # Decoder
        self.decoder_input_layer = InputLayer(
            args["num_classes"],
            args["model_dim"],
            args["max_seq_len"],
            args["input_dropout"],
            self.device,
        )
        self.decoder_stack = nn.ModuleList(
            [
                DecoderLayer(
                    args["model_dim"],
                    args["keys_dim"],
                    args["values_dim"],
                    args["heads"],
                    self.device,
                )
                for _ in range(args["decoder_stack_size"])
            ]
        )

        self.output_layer = nn.Linear(
            args["model_dim"], args["num_classes"], device=self.device
        )

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        max_seq_len: int = 50,
    ) -> torch.Tensor:
        """
        Assume 0 is <BOS> and 1 is <EOS> token.
        """
        # Encoder
        encoder_outputs = self.encoder_input_layer(encoder_inputs)
        encoder_outputs = self.encoder_stack(encoder_outputs)

        # Decoder
        if targets is not None:
            # TRAIN: teacher forcing with self_att causal mask
            decoder_outputs = self.decoder_input_layer(targets)
            for decoder in self.decoder_stack:
                decoder_outputs = decoder(decoder_outputs, encoder_outputs, True)
            out = self.out_layer(decoder_outputs)
        else:
            # INFERENCE: auto-regressive decoding
            batch_size = encoder_inputs.size(0)
            decoder_inputs = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )
            out = self.greedy_search(max_seq_len, encoder_outputs, decoder_inputs)

        return out

    def _decoder_stack_forward(self, encoder_outputs, decoder_inputs):
        decoder_outputs = self.decoder_input_layer(decoder_inputs)
        for decoder in self.decoder_stack:
            decoder_outputs = decoder(decoder_outputs, encoder_outputs, True)
        out = self.output_layer(decoder_outputs)
        return out

    def greedy_search(
        self,
        max_seq_len: int,
        encoder_outputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ):
        """
        Simple greedy top 1 search and auto-regressive decoding. Output is a tensor of
        shape (B,T,F).
        """
        for _ in range(max_seq_len):
            out = self._decoder_stack_forward(encoder_outputs, decoder_inputs)
            _, topi = out[:, -1].topk(1)
            next_tokens = topi.detach()
            decoder_inputs = torch.cat((decoder_inputs, next_tokens), dim=1)

        return out
