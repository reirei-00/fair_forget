"""Copied from https://github.com/centerforaisafety/wmdp."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rmu.config import resolve_model


################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_decoder_layers(model):
    """Return the list of decoder layers.

    Text-only models (Llama, Qwen) keep them at ``model.model.layers``; multimodal Gemma 3
    checkpoints (Lapa, Gemma, Mamay) at ``model.model.language_model.layers``.
    """
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers


def get_hidden_size(model):
    # Multimodal configs keep the text model's settings in text_config.
    return model.config.get_text_config().hidden_size


def get_params(model, layer_ids, param_names):
    """Select parameters by name (e.g. ``mlp.down_proj.weight``) in the given layers.

    Names are used instead of positions because the parameter order differs between
    architectures: ``down_proj`` is parameter 6 in a Llama layer but 8 in a Gemma 3 layer.
    """
    layers = get_decoder_layers(model)
    params = []
    for layer_id in layer_ids:
        named = dict(layers[layer_id].named_parameters())
        for name in param_names:
            if name not in named:
                raise ValueError(
                    f"layer {layer_id} has no parameter {name!r}; available: {list(named)}"
                )
            params.append(named[name])
    return params


def load_model(model_name_or_path):
    # Short names from config/default.yaml (e.g. "lapa-12b-instruct") map to the shared models dir.
    model_name_or_path = resolve_model(model_name_or_path)
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        # "auto" spreads big models over the GPUs; without CUDA it can hang, so load on CPU.
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer
