# الملف: model_runner_node.py (النسخة النهائية المصححة)

import requests
import json
import torch
import numpy as np
from PIL import Image
import base64
import io

class AtyImageDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "language": (["Arabic", "English", "Spanish", "French", "Japanese"], {"default": "Arabic"}),
                "max_tokens": ("INT", {"default": 250, "min": 50, "max": 1024, "step": 10}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe_image"
    CATEGORY = "Aty Tools"

    def pil_to_base64(self, pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def describe_image(self, image: torch.Tensor, language: str, max_tokens: int):
        model_filename = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
        api_url = "http://localhost:4891/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        try:
            img_tensor = image[0]
            img_np = np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            base64_image = self.pil_to_base64(pil_image)

            # --- هذا هو الحل الصحيح الذي تم التوصل إليه ---
            # 1. النص يحتوي على مؤشر للصورة مثل [img-10]
            user_prompt_text = f"Describe this image in {language}. \n[img-10]"
            
            # 2. الحمولة تفصل بين الرسالة النصية وبيانات الصورة
            payload = {
                "model": model_filename,
                "messages": [{"role": "user", "content": user_prompt_text}],
                "image_data": [{"data": base64_image, "id": 10}],
                "temperature": 0.2,
                "max_tokens": max_tokens
            }
            
            print("Aty_Desc_Node: Sending request with CORRECTED payload format...")
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            data = response.json()
            description = data['choices'][0]['message']['content'].strip()
            print(f"Aty_Desc_Node: Received description: {description}")

        except requests.exceptions.HTTPError as err:
            error_message = f"HTTP ERROR: {err.response.status_code}. Response: {err.response.text}"
            print(error_message)
            return {"ui": {"string": [error_message]}, "result": (error_message,)}
        except Exception as e:
            error_message = f"GENERAL ERROR in Aty Node: {e}"
            print(error_message)
            return {"ui": {"string": [error_message]}, "result": (error_message,)}
            
        # الآن سيعمل هذا الجزء ويعرض الوصف في المربع
        return {"ui": {"string": [description]}, "result": (description,)}

# الأجزاء السفلية تبقى كما هي
NODE_CLASS_MAPPINGS = {"AtyImageDescriber": AtyImageDescriber}
NODE_DISPLAY_NAME_MAPPINGS = {"AtyImageDescriber": "Aty Image Describer"}