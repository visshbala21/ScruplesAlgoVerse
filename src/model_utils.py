import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model(model_id):
    """
    Load and return the model and tokenizer.
    
    Args:
    model_id (str): The model identifier from Hugging Face.

    Returns:
    model: The loaded model.
    tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
    return model, tokenizer

def get_model_probabilities(text, model, tokenizer, device='cuda'):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Focus on the logits of the last token in the sequence
    last_token_logits = logits[:, -1, :]
    probs = torch.softmax(last_token_logits, dim=-1)
    return last_token_logits, probs
