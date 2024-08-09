import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.config import load_config

def load_model_from_config(config_path):
    """
    Load the model and tokenizer based on the given configuration file.

    Args:
    config_path (str): Path to the YAML config file.

    Returns:
    model: The loaded model.
    tokenizer: The loaded tokenizer.
    """
    config = load_config(config_path)
    model_id = config['model_id']
    quantization_config = BitsAndBytesConfig(**config['quantization_config'])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    return model, tokenizer
