import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os


def llama2_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama3_apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False,
                                           enable_thinking=False)
    encoded = tokenizer(prompt)
    return encoded


def qwen2_apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False,
                                           enable_thinking=False)
    encoded = tokenizer(prompt)
    return encoded


def qwen3_apply_chat_template(prompt, tokenizer):
    """Format a Qwen3 prompt with thinking mode disabled."""
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False,
                                           enable_thinking=False)
    encoded = tokenizer(prompt)
    return encoded


def get_model_type_arch(model_name_or_path):
    if any([
            x in model_name_or_path.lower()
            for x in ["llama-2", "llama2", "llama_2"]
    ]):
        print("run llama2 model")
        return "llama2", "llama"
    elif any([
            x in model_name_or_path.lower() for x in
        ["llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3", "llama_3"]
    ]):
        print("run llama3 model")
        return "llama3", "llama"
    elif any([x in model_name_or_path.lower() for x in ["qwen2", "qwen2.5"]]):
        print("run qwen2 model")
        return "qwen2", "qwen2"
    elif any([x in model_name_or_path.lower() for x in ["qwen3", "qwen3.0"]]):
        print("run qwen3 model")
        return "qwen3", "qwen3"
    elif any([x in model_name_or_path.lower() for x in ["mistral"]]):
        print("run mistral model")
        return "mistral", "llama"
    else:
        raise ValueError("Unsupported model name")


def load_config_and_tokenizer(args, task_config, model_name_or_path):
    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    model_config.torch_dtype = dtype
    generate_kwargs = {}

    method = args.method.lower()

    model_type, model_arch = get_model_type_arch(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True,
                                              device_map={"": 0},
                                              use_fast=True)

    if model_type == "llama2":
        apply_chat_template = llama2_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif model_type in ["llama3", "mistral"]:
        print("run llama3/mistral model")
        apply_chat_template = llama3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    elif model_type == "qwen2":
        print("run qwen2 model")
        apply_chat_template = qwen2_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    elif model_type == "qwen3":
        print("run qwen3 model with thinking disabled")
        apply_chat_template = qwen3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    else:
        raise ValueError("Unsupported model name")

    if method == "flashattn":
        generate_config = {
            "max_gpu_cache_memory":
            float(task_config.get('device', 'CUDA_MEM')) *
            1024 * 1024 * 1024,
        }

    elif method == "hash":
        print("run sparse hash model")
        generate_config = {
            "max_gpu_cache_memory":
            float(task_config.get('device', 'CUDA_MEM')) * 1024 * 1024 * 1024,
            "hash_rbits": int(task_config.get('method', 'RBIT')),
            "hash_weights_path": os.environ.get(
                "HHR_WEIGHTS_PATH",
                task_config.get('method', 'HASH_WEIGHTS_PATH'),
            ),
            "sparse_ratio": float(task_config.get('dataset', 'TOPK_RATIO')),
            "with_bias": False,
            "num_sink": int(task_config.get('method', 'NUM_SINK')),
            "num_recent": int(task_config.get('method', 'NUM_RECENT')),
            "quest_ratio": float(task_config.get("quest", "RATIO", fallback=0.0)),
            "quest_page_size": int(task_config.get("quest", "PAGE_SIZE", fallback=16)),
            "use_hadamard": task_config.getboolean(
                "method", "USE_HADAMARD", fallback=False
            ),
        }
        print("hash config: ", generate_config)

    else:
        raise ValueError(f"Unsupported method: {method}")

    model_meta = (method, model_arch, generate_config)

    return model_meta, model_config, tokenizer, generate_kwargs, apply_chat_template


def load_model(model_meta, model_config, model_name_or_path):
    method, model_arch, generate_config = model_meta

    if method == "flashattn":  
        if model_arch == "llama":
            # Full attention uses the standard Transformers implementation.
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=model_config,
            )
        elif model_arch == "qwen2":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=model_config,
                attn_implementation="flash_attention_2"
            )
        elif model_arch == "qwen3":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=model_config,
                attn_implementation="flash_attention_2"
            )
        else:
            raise NotImplementedError(
                f"{method} not implemented for {model_arch} models!")

    elif method == "hash":
        if model_arch == "llama":
            print("run sparse hash Llama model")
            from myTransformer.models.llama.modeling_llama_hash import CustomLlamaForCausalLM
            ## Mistral config compatibility
            for attr in ["attention_bias", "mlp_bias"]:
                if not hasattr(model_config, attr):
                    setattr(model_config, attr, False)
            if not hasattr(model_config, "rope_scaling"):
                model_config.rope_scaling = None
            model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
          
        elif model_arch == "qwen2":
            from myTransformer.models.qwen2.modeling_qwen2_hash import CustomQwen2ForCausalLM
            model = CustomQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
        elif model_arch == "qwen3":
            from myTransformer.models.qwen2.modeling_qwen3_hash import CustomQwen3ForCausalLM
            model = CustomQwen3ForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
        else:
            raise NotImplementedError(
                f"{method} not implemented for {model_arch} models!")

    else:
        raise ValueError(f"Unsupported method: {method}")

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    dtype = torch.float16
    model = model.to(dtype).eval()

    for key, value in generate_config.items():
        setattr(model.generation_config, key, value)

    return model


def comm_generate(x, generate_kwarg, model, tokenizer):
    input_length = x["input_ids"].shape[1]

    output = model.generate(**x, do_sample=False, **generate_kwarg)

    output = output[:, input_length:]

    preds = tokenizer.batch_decode(output, skip_special_tokens=True)
    # A byte-level token may be cut at max_new_tokens. Remove only a trailing\n    # incomplete UTF-8 replacement marker; valid decoded text is untouched.\n    preds = [p[:-1].rstrip() if p.endswith('\ufffd') else p for p in preds]\n    # Strip Qwen3 thinking tokens from output
    preds = [p.replace('<think>\n', '').replace('</think>', '').strip() for p in preds]
    return preds
