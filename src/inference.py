import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.special import softmax
import numpy as np

def get_yes_no_probabilities(probs, tokenizer):
    try:
        yes_token_id = tokenizer.convert_tokens_to_ids('yes')
        no_token_id = tokenizer.convert_tokens_to_ids('no')
        Yes_token_id = tokenizer.convert_tokens_to_ids('Yes')
        No_token_id = tokenizer.convert_tokens_to_ids('No')

        yes_prob = probs[0, yes_token_id].item() + probs[0, Yes_token_id].item()
        no_prob = probs[0, no_token_id].item() + probs[0, No_token_id].item()
    except IndexError:
        yes_prob = 0.0
        no_prob = 0.0

    return yes_prob, no_prob

def calculate_kl_divergence(model_probs, human_probs):
    return entropy(model_probs, human_probs)

def dirichlet_multinomial_loss(pred_probs, counts, alpha=1.0):
    pred_probs = torch.tensor(pred_probs, dtype=torch.float32)
    counts = torch.tensor(counts, dtype=torch.float32)
    alpha = alpha * pred_probs
    log_probs = torch.lgamma(alpha + counts) - torch.lgamma(alpha)
    log_probs = log_probs.sum(dim=-1) - (torch.lgamma(alpha.sum(dim=-1) + counts.sum(dim=-1)) - torch.lgamma(alpha.sum(dim=-1)))
    return -log_probs.mean().item()

def xentropy(human_probs, model_probs):
    human_probs_tensor = torch.tensor(human_probs, dtype=torch.float32)
    model_probs_tensor = torch.tensor(model_probs, dtype=torch.float32)
    loss = F.cross_entropy(model_probs_tensor.unsqueeze(0), human_probs_tensor.unsqueeze(0))
    return loss.item()
