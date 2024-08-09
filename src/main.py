from src.model_utils import load_model
from src.data_utils import load_jsonl_data, sample_data, save_to_csv
from src.evaluation import evaluate_scenarios
from src.visualization import plot_distributions

def main():
    # Load the model
    model_id = "meta-llama/Meta-Llama-3-8B"
    model, tokenizer = load_model(model_id)
    
    # Load and sample data
    dilemmas_df = load_jsonl_data("/content/train.scruples-dilemmas.jsonl")
    sampled_dilemmas = sample_data(dilemmas_df)
    save_to_csv(sampled_dilemmas, "sampled_dilemmas.csv")
    
    # Evaluate scenarios
    results_df = evaluate_scenarios(sampled_dilemmas, model, tokenizer)
    save_to_csv(results_df, "results.csv")
    
    # Generate visualizations
    plot_distributions(results_df)
    
if __name__ == "__main__":
    main()
