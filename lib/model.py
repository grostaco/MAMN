import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from layers.atttention import IntraAttentionStack, GlobalAttention
from .utils import wdmc


class MAMNConfig(PretrainedConfig):
    model_type = 'MAMN'

    def __init__(self, num_embeddings: int = 30522, d_hidden: int = 768, num_heads: int = 4,
                 layers: int = 4, window_size: int = 8, **kwargs):
        self.num_embeddings = num_embeddings
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.layers = layers
        self.window_size = window_size

        super().__init__(**kwargs)


class MAMN(PreTrainedModel):
    config_class = MAMNConfig

    def __init__(self, config: MAMNConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.num_embeddings, config.d_hidden)
        self.intra_attn_layers1 = IntraAttentionStack(
            config.d_hidden, config.num_heads, config.layers)
        self.intra_attn_layers2 = IntraAttentionStack(
            config.d_hidden, config.num_heads, config.layers)

        self.window_size = config.window_size
        self.global_attn1 = GlobalAttention(config.d_hidden, config.d_hidden)
        self.global_attn2 = GlobalAttention(config.d_hidden, config.d_hidden)

        self.dense = nn.Linear(config.d_hidden, config.num_labels)

        self.config = config

    def forward(self, input_ids: torch.LongTensor,
                aspects_input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                aspects_attention_mask: torch.LongTensor,
                start: tuple[int, ...],
                end: tuple[int, ...],
                labels=None,
                **kwargs):
        context_emb = self.embedding(input_ids)
        aspects_emb = self.embedding(aspects_input_ids)

        h_cp = self.intra_attn_layers1(context_emb, attn_mask=attention_mask.bool(
        ) if attention_mask is not None else None)
        h_ap = self.intra_attn_layers2(
            aspects_emb, attn_mask=aspects_attention_mask.bool() if attention_mask is not None else None)

        h_cw = wdmc(h_cp, start, end, self.config.window_size)

        g = self.global_attn1(h_cw, h_ap)  # * aspects_attention_mask

        h_cw_avg = torch.mean(h_cw, dim=1)  # * attention_mask

        attn_weights = self.global_attn2(h_ap, h_cw_avg, only_weights=True)

        O = (attn_weights @ g)
        logits = F.tanh(self.dense(O))[:, 0]

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=attn_weights
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear | GlobalAttention):
            torch.nn.init.uniform(module.weight)
