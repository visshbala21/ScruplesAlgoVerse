import pandas as pd
import numpy as np
from src.model_utils import get_model_probabilities
from src.inference import get_yes_no_probabilities, xentropy, dirichlet_multinomial_loss

def evaluate_anecdotes(df, model, tokenizer, few_shot_prompt):
    scenarios = []
    yes_probabilities = []
    no_probabilities = []
    human_right_probabilities = []
    human_wrong_probabilities = []
    xentropies = []
    dirichlet_losses = []
    human_responses = []
    binarized_labels = []

    for idx, row in df.iterrows():
        scenario = row['text']
        input_text = few_shot_prompt.format(scenario)

        logits, probs = get_model_probabilities(input_text, model, tokenizer)
        human_response = row['binarized_label_scores']

        yes_prob, no_prob = get_yes_no_probabilities(probs, tokenizer)

        scenarios.append(input_text)
        yes_probabilities.append(yes_prob * 100)
        no_probabilities.append(no_prob * 100)
        human_responses.append(human_response)

        total_responses = sum(human_response.values())
        human_right_prob = human_response['RIGHT'] / total_responses
        human_wrong_prob = human_response['WRONG'] / total_responses

        human_right_probabilities.append(human_right_prob * 100)
        human_wrong_probabilities.append(human_wrong_prob * 100)

        model_probs = np.array([yes_prob, no_prob])
        human_probs = np.array([human_right_prob, human_wrong_prob])
        xentropy_value = xentropy(human_probs, model_probs)
        xentropies.append(xentropy_value)

        dirichlet_loss = dirichlet_multinomial_loss(model_probs, [human_response['RIGHT'], human_response['WRONG']])
        dirichlet_losses.append(dirichlet_loss)

        binarized_label_str = 'RIGHT' if human_right_prob > human_wrong_prob else 'WRONG'
        binarized_labels.append(binarized_label_str)

    results_df = pd.DataFrame({
        'scenario': scenarios,
        'yes_probability': yes_probabilities,
        'no_probability': no_probabilities,
        'human_response': human_responses,
        'binarized_label': binarized_labels
    })

    return results_df
