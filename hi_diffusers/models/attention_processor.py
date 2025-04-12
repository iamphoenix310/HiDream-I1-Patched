from typing import Optional
import torch
from .attention import HiDreamAttention
from .attention_processor_torch import HiDreamAttnProcessor_torch

# ⚠️ Patch: Disable flash-attn by aliasing torch-based fallback
HiDreamAttnProcessor_flashattn = HiDreamAttnProcessor_torch
