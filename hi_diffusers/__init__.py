# hi_diffusers/__init__.py

# Only import core interfaces, not deep dependencies that may recurse
from .pipelines.hidream_image import pipeline_hidream_image
from .models.transformers import transformer_hidream_image
