# -*- coding: utf-8 -*-

def get_model_type_arch(model_path: str):
    """
    Return:
        model_name: llama3 / llama2 / longchat / qwen2 / glm
        model_arch: llama / qwen2 / glm

    Used by weights/build_dataset.py.
    """

    name = str(model_path).lower()

    # Llama family
    if "llama-3" in name or "llama3" in name or "llama-3.1" in name or "llama-3.2" in name:
        return "llama3", "llama"

    if "llama-2" in name or "llama2" in name:
        return "llama2", "llama"

    if "longchat" in name:
        return "longchat", "llama"

    # Qwen2 family
    if "qwen2" in name or "qwen-2" in name or "qwen2.5" in name or "qwen-2.5" in name:
        return "qwen2", "qwen2"

    # GLM family
    if "glm" in name:
        return "glm", "glm"

    # Default to the Llama-style implementation.
    print(f"[Warning] Cannot infer model type from path: {model_path}")
    print("[Warning] Default to model_name='llama3', model_arch='llama'")
    return "llama3", "llama"
