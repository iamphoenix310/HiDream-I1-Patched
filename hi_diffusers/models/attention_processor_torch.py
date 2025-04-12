from typing import Optional
import torch
from .attention import HiDreamAttention

class HiDreamAttnProcessor_torch:
    def __call__(
        self,
        attn: HiDreamAttention,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)

        if image_tokens_masks is not None:
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        attn_scores = torch.einsum("bqhd,bkhd->bhqk", query_i, key_i) / (head_dim**0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, value_i)

        hidden_states = attn_output.flatten(2)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states
