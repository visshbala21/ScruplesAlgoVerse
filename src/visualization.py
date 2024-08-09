import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distributions(df):
    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['yes_probabilities'], kde=True, color='blue')
    plt.title('Distribution of Yes Probabilities')
    plt.xlabel('Probability (%)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df['no_probabilities'], kde=True, color='red')
    plt.title('Distribution of No Probabilities')
    plt.xlabel('Probability (%)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['human_right_probabilities'], kde=True, color='green')
    plt.title('Distribution of Human Right Probabilities')
    plt.xlabel('Probability (%)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df['human_wrong_probabilities'], kde=True, color='orange')
    plt.title('Distribution of Human Wrong Probabilities')
    plt.xlabel('Probability (%)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
    # Add more plots as needed
