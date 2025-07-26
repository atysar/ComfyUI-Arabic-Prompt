import requests
import json

class ArabicPromptToConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text_arabic": ("STRING", {
                    "multiline": True,
                    "default": "ناطحة سحاب في مدينة مزدحمة ليلاً"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive_conditioning",)
    FUNCTION = "translate_and_encode"
    CATEGORY = "Prompt Tools"

    def translate_and_encode(self, clip, text_arabic):
        # تم التأكد من أن هذا هو النموذج الصحيح
        model_filename = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
        
        api_url = "http://localhost:4891/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # ==================================================================
        #  == تم إعادة كتابة التعليمات لتكون صارمة ومناسبة لـ Llama 3.1 ==
        # ==================================================================
        system_prompt = (
            "You are a direct, one-job translation machine. "
            "Your ONLY function is to translate the user's text from Arabic to English. "
            "The output MUST be ONLY the translated English text, and nothing else. "
            "DO NOT add any conversational phrases, introductions, or explanations. "
            "For example, if the input is 'سيارة حمراء', the output MUST be 'a red car'."
        )

        payload = {
            "model": model_filename,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_arabic}
            ],
            "temperature": 0.0,  # تم ضبط درجة الحرارة إلى الصفر لمنع أي إبداع
            "max_tokens": 150   # نحدد له حدًا أقصى للكلمات لمنعه من الإطالة
        }

        try:
            print("Llama3.1_Node: Sending STRICT translation request...")
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=45)
            response.raise_for_status()
            data = response.json()
            
            translated_text = data['choices'][0]['message']['content'].strip()
            print(f"Llama3.1_Node: Received response: {translated_text}")

        except Exception as e:
            error_message = f"ERROR connecting to GPT4All: {e}"
            print(error_message)
            return ([],)

        print("Llama3.1_Node: Encoding the translated text...")
        tokens = clip.tokenize(translated_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        return ([[cond, {"pooled_output": pooled}]], )

NODE_CLASS_MAPPINGS = {
    "ArabicPromptToConditioning": ArabicPromptToConditioning
}

# سنحتفظ بنفس الاسم
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArabicPromptToConditioning": "Arabic Prompt PL"
}