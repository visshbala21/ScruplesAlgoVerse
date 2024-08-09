import pandas as pd

def load_jsonl_data(file_path):
    """
    Load a JSONL file into a pandas DataFrame.
    
    Args:
    file_path (str): Path to the JSONL file.
    
    Returns:
    DataFrame: Loaded DataFrame.
    """
    df = pd.read_json(file_path, lines=True)
    return df

def sample_data(df, frac=0.01, random_state=42):
    """
    Sample a fraction of the dataset for quicker processing.
    
    Args:
    df (DataFrame): The original DataFrame.
    frac (float): Fraction of the dataset to sample.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    DataFrame: Sampled DataFrame.
    """
    return df.sample(frac=frac, random_state=random_state)

def save_to_csv(df, output_path):
    """
    Save a DataFrame to a CSV file.
    
    Args:
    df (DataFrame): DataFrame to save.
    output_path (str): Path where the CSV should be saved.
    """
    df.to_csv(output_path, index=False)
