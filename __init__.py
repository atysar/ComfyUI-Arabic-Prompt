# الملف: __init__.py

from .arabic_prompt_node import ArabicPromptToConditioning
from .model_runner_node import AtyImageDescriber

NODE_CLASS_MAPPINGS = {
    "AtyArabicPromptToConditioning": ArabicPromptToConditioning,
    "AtyImageDescriber": AtyImageDescriber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AtyArabicPromptToConditioning": "Aty Arabic Prompt PL",
    "AtyImageDescriber": "Aty Image Describer",
}

print("✅ Loaded 'Aty Llama Nodes' package")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']