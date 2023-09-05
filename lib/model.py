import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.atttention import IntraAttentionStack, GlobalAttention
from .utils import wdmc


class MAMN(nn.Module):
    def __init__(self, num_embeddings: int, d_hidden: int, num_heads: int, layers: int, window_size: int,
                 num_labels: int):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, d_hidden)
        self.intra_attn_layers = IntraAttentionStack(
            d_hidden, num_heads, layers)

        self.window_size = window_size
        self.global_attn1 = GlobalAttention(d_hidden, d_hidden)
        self.global_attn2 = GlobalAttention(d_hidden, d_hidden)

        self.dense = nn.Linear(d_hidden, num_labels)

    def forward(self, context: torch.LongTensor,
                aspects: torch.LongTensor,
                a_ranges: tuple[tuple[int, int], ...]):
        context_emb = self.embedding(context)
        aspects_emb = self.embedding(aspects)

        h_cp = self.intra_attn_layers(context_emb)
        h_ap = self.intra_attn_layers(aspects_emb)

        h_cw = wdmc(h_cp, a_ranges, self.window_size)
        g = self.global_attn1(h_cw, h_ap)

        h_cw_avg = torch.mean(h_cw, dim=1)

        attn_weights = self.global_attn2(h_cw_avg, h_ap, only_weights=True)

        O = (attn_weights @ g).squeeze(1)
        logits = F.tanh(self.dense(O))

        return F.softmax(logits, -1)
