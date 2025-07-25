import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import os
import concurrent.futures
import comfy.utils

# =================================================================================
# 1. HELPER FUNCTIONS
# =================================================================================

def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)

def pil_to_base64(pil_image, format="JPEG"):
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODELS_CACHE = None

def get_ollama_models(url="http://127.0.0.1:11434/api/tags"):
    """
    Dynamically fetches the list of local Ollama models.
    Returns a default list if the API call fails, to prevent ComfyUI from crashing.
    """
    global OLLAMA_MODELS_CACHE
    if OLLAMA_MODELS_CACHE is not None:
        return OLLAMA_MODELS_CACHE
        
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        # Prefer vision models, but fall back to all models if none are found
        model_names = sorted([model["name"] for model in models if "vision" in model.get("details", {}).get("families", []) or "vl" in model.get("name", "")])
        if not model_names:
            print("[OllamaVisionKit] Warning: No recommended vision models (e.g., llava, qwen:vl) found. Displaying all available models.")
            model_names = sorted([model["name"] for model in models])

        OLLAMA_MODELS_CACHE = model_names
        return model_names
    except Exception as e:
        print(f"[OllamaVisionKit] Error: Could not connect to Ollama API at {url} to fetch models: {e}")
        print("[OllamaVisionKit] Info: Using a default model list. Please ensure the Ollama service is running.")
        return ["llava:latest", "qwen2.5vl:7b", "moondream:latest"]

def call_ollama_api(image_b64, model_name, instruction, url):
    payload = {"model": model_name, "prompt": instruction, "images": [image_b64], "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("response", "").strip().replace('"', '')
    except Exception as e:
        return f"Ollama API Error: {e}"

def process_image_task(image_path, output_dir, prompt, trigger_word, model, url):
    """
    A single image processing task designed for concurrent execution.
    """
    filename = os.path.basename(image_path)
    try:
        pil_image = Image.open(image_path).convert("RGB")
        b64_image = pil_to_base64(pil_image)
        
        caption = call_ollama_api(b64_image, model, prompt, url)
        
        if "Ollama API Error" in caption:
            return (filename, "API_ERROR", caption)
        
        final_text = f"{trigger_word}, {caption}" if trigger_word else caption
        base_filename, _ = os.path.splitext(filename)
        txt_filepath = os.path.join(output_dir, f"{base_filename}.txt")
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(final_text)
            
        return (filename, "SUCCESS", caption[:80] + "...") # Return a summary for logging
    except Exception as e:
        return (filename, "FILE_ERROR", str(e))

# =================================================================================
# 2. NODE DEFINITIONS
# =================================================================================

class OllamaAdvancedConfigurator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zero_tolerance_mode": ("BOOLEAN", {"default": True}),
                "output_style": (['Natural Language Paragraph', 'Comma-Separated Tags'],),
                "description_length": (['Concise', 'Medium', 'Detailed'],),
            },
            "optional": { "content_options": ("OLLAMA_CONTENT_OPTIONS",) }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instruction",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Ollama"
    NODE_DISPLAY_NAME = "Ollama Advanced Configurator"

    def generate_prompt(self, zero_tolerance_mode, output_style, description_length, content_options=None):
        # Default options if nothing is connected
        if content_options is None:
            content_options = {
                "include_subject": True, "include_hair": True, "include_clothing": True, "include_accessories": True,
                "include_pose": True, "include_background": True, "include_style": True, "include_lighting": True,
                "include_composition": True, "include_quality": True
            }

        prompt = ""
        if zero_tolerance_mode:
            prompt += "ZERO-TOLERANCE POLICY ON HALLUCINATION & OMISSION:\nYour primary directive is absolute factual accuracy. You are strictly forbidden from inventing or assuming details not clearly visible. It is better to omit an uncertain detail than to describe it incorrectly.\n\n"
        
        prompt += "Your Task:\nGenerate a high-quality caption for this image.\n\n"
        prompt += "Analysis Checklist:\n"
        
        if content_options.get('include_subject'): prompt += "- Subject's identity (gender, race if apparent).\n"
        if content_options.get('include_hair'): prompt += "- Hair details (style, color, length).\n"
        if content_options.get('include_clothing'): prompt += "- Clothing and layers.\n"
        if content_options.get('include_accessories'): prompt += "- All accessories (bags, jewelry, etc.).\n"
        if content_options.get('include_pose'): prompt += "- Pose and facial expression.\n"
        if content_options.get('include_background'): prompt += "- Setting and background.\n"
        if content_options.get('include_style'): prompt += "- Overall artistic style (e.g., photography type).\n"
        if content_options.get('include_lighting'): prompt += "- Lighting conditions.\n"
        if content_options.get('include_composition'): prompt += "- Composition (shot type, depth of field).\n"
        if content_options.get('include_quality'): prompt += "- Image quality (e.g., masterpiece, high quality).\n"

        prompt += "\nFinal Output Mandate:\n"
        
        # Maps must use the English keys from the dropdown menu
        length_map = {
            'Concise': 'Be concise.', 
            'Medium': 'Provide a balanced level of detail.', 
            'Detailed': 'Be highly detailed. Embrace detail. The more accurate details you provide, the better.'
        }
        style_map = {
            'Natural Language Paragraph': "Synthesize your analysis into one fluid, descriptive paragraph. Start directly with the subject. Do not mention the checklist.",
            'Comma-Separated Tags': "Provide your analysis as a list of comma-separated keywords or phrases."
        }
        
        prompt += style_map.get(output_style, "") + " "
        prompt += length_map.get(description_length, "") + "\n"
        
        prompt += "Output ONLY the final caption in English."
        return (prompt,)

class OllamaContentOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "include_subject": ("BOOLEAN", {"default": True}), "include_hair": ("BOOLEAN", {"default": True}),
            "include_clothing": ("BOOLEAN", {"default": True}), "include_accessories": ("BOOLEAN", {"default": True}),
            "include_pose": ("BOOLEAN", {"default": True}), "include_background": ("BOOLEAN", {"default": True}),
            "include_style": ("BOOLEAN", {"default": True}), "include_lighting": ("BOOLEAN", {"default": True}),
            "include_composition": ("BOOLEAN", {"default": True}), "include_quality": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("OLLAMA_CONTENT_OPTIONS",)
    FUNCTION = "package_options"
    CATEGORY = "Ollama"
    NODE_DISPLAY_NAME = "Ollama Content Options"
    
    def package_options(self, **kwargs):
        return (kwargs,)

class OllamaInterrogator:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { 
            "image": ("IMAGE",), 
            "instruction": ("STRING", {"forceInput": True}), 
            "model": (get_ollama_models(),),
            "ollama_url": ("STRING", {"default": OLLAMA_API_URL}), 
        } }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "interrogate"
    CATEGORY = "Ollama"
    NODE_DISPLAY_NAME = "Ollama Interrogator"

    def interrogate(self, image, instruction, model, ollama_url):
        b64_img = pil_to_base64(tensor_to_pil(image))
        return (call_ollama_api(b64_img, model, instruction, ollama_url),)

class OllamaBatchTagger:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { 
            "instruction": ("STRING", {"forceInput": True}),
            "image_folder": ("STRING", {"default": "C:\\ComfyUI\\input"}),
            "caption_folder": ("STRING", {"default": "C:\\ComfyUI\\output\\captions"}),
            "trigger_word": ("STRING", {"default": "my_trigger_word"}),
            "model": (get_ollama_models(),),
            "ollama_url": ("STRING", {"default": OLLAMA_API_URL}),
            }, 
            "optional": { 
                "clear_output_directory": ("BOOLEAN", {"default": True}),
                "concurrency": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
            } 
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "batch_tag"
    CATEGORY = "Ollama"
    NODE_DISPLAY_NAME = "Ollama Batch Tagger"
    OUTPUT_NODE = True
    
    def batch_tag(self, instruction, image_folder, caption_folder, trigger_word, model, ollama_url, clear_output_directory=True, concurrency=4):
        if not os.path.isdir(image_folder):
            return (f"Error: Input folder does not exist: {image_folder}",)
        
        os.makedirs(caption_folder, exist_ok=True)
        if clear_output_directory:
            print(f"[OllamaVisionKit] Clearing output directory: {caption_folder}")
            for f in os.listdir(caption_folder):
                if f.endswith(".txt"):
                    os.remove(os.path.join(caption_folder, f))
        
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        total = len(image_files)
        
        if total == 0:
            msg = f"No supported image files found in the folder: {image_folder}"
            print(f"[OllamaVisionKit] {msg}")
            return (msg,)
            
        print(f"[OllamaVisionKit] Found {total} images. Starting processing with {concurrency} workers...")
        
        pbar = comfy.utils.ProgressBar(total)
        success_count, error_count = 0, 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_image_task, img_path, caption_folder, instruction, trigger_word, model, ollama_url) for img_path in image_files]
            
            for future in concurrent.futures.as_completed(futures):
                filename, status, message = future.result()
                if status == "SUCCESS":
                    print(f"  [SUCCESS] {filename} -> {message}")
                    success_count += 1
                else:
                    print(f"  [ERROR] {filename} -> {message}")
                    error_count += 1
                pbar.update(1)

        summary = f"Batch processing complete. Successful: {success_count}, Failed: {error_count}. Captions saved to {caption_folder}."
        print(f"--- {summary} ---")
        return (summary,)

# =================================================================================
# 3. NODE MAPPINGS
# =================================================================================
NODE_CLASS_MAPPINGS = {
    "OllamaAdvancedConfigurator": OllamaAdvancedConfigurator,
    "OllamaContentOptions": OllamaContentOptions,
    "OllamaInterrogator": OllamaInterrogator,
    "OllamaBatchTagger": OllamaBatchTagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaAdvancedConfigurator": "Ollama Advanced Configurator",
    "OllamaContentOptions": "Ollama Content Options",
    "OllamaInterrogator": "Ollama Interrogator",
    "OllamaBatchTagger": "Ollama Batch Tagger",
}