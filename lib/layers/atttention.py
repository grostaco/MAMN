import torch
import torch.nn as nn
import torch.nn.functional as F

from .pffn import PointWiseFFN


class IntraAttentionStack(nn.Module):
    def __init__(self, d_hidden: int, num_heads: int, layers: int):
        super().__init__()

        self.layers = nn.ModuleList([
            IntraAttentionBlock(d_hidden, d_hidden, num_heads)
            for _ in range(layers)
        ])

    def forward(self, emb: torch.Tensor):
        for layer in self.layers:
            emb = layer(emb)

        return emb


class IntraAttentionBlock(nn.Module):
    def __init__(self, d_hidden: int, d_ff: int, num_heads: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_hidden, num_heads, dropout=.2, batch_first=True)
        self.pffn = PointWiseFFN(d_hidden, d_ff)

    def forward(self, emb: torch.Tensor):
        attn_output, _ = self.attn(emb, emb, emb, need_weights=False)
        h = self.pffn(attn_output)

        return h


class GlobalAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    # TODO: implement biases
    def forward(self, h_cw: torch.Tensor, h_ap: torch.Tensor, only_weights=False):
        logits = F.tanh(h_cw @ self.weights @
                        h_ap.swapaxes(1, 2))  # + self.bias

        I_attn = F.softmax(logits,
                           dim=-1)

        if only_weights:
            return I_attn.squeeze(1)
        return I_attn @ h_cw

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
