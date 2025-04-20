import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

with open('experiment/experiment_results/debias_experiment_results.json', 'r') as file:
    data = json.load(file)
    
random_sources_biases = []
random_response_biases = []
selected_sources_biases = []
selected_response_biases = []

for prompt, dict in data.items():
    random_sources_biases.append(dict["unbalanced_source_bias_score"])
    random_response_biases.append(dict["unbalanced_response_bias_score"])
    selected_sources_biases.append(dict["balanced_source_bias_score"])
    selected_response_biases.append(dict["balanced_response_bias_score"])

def plot_bias_scores(sources_biases, response_biases, balanced: bool):
    # Calculate correlation and regression line
    slope, intercept, r_value, _, _ = linregress(sources_biases, response_biases)
    regression_line = [slope * x + intercept for x in sources_biases]
    absolute_bias_score = sum([abs(x) for x in response_biases]) / len(response_biases)

    # Plot total_source_bias_score vs output_bias_score
    plt.figure(figsize=(10, 6))
    plt.scatter(sources_biases, response_biases, color='blue', alpha=0.7, label='Data Points')
    plt.plot(sources_biases, regression_line, color='red', label=f'Correlation Line (r={r_value:.2f})')
    # Add title, labels, and legend
    title_suffix = " (Balanced)" if balanced else " (Random)"
    plt.title("Total Source Bias Score vs Output Bias Score" + title_suffix, fontsize=14)
    plt.xlabel("Total Source Bias Score", fontsize=12)
    plt.ylabel("Output Bias Score", fontsize=12)
    plt.legend(fontsize=10, title=f"Absolute Avg. Bias Score: {absolute_bias_score:.2f}")

    # Add grid and save the plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'experiment/experiment_results/debias_experiment_scatter_plot_{"balanced" if balanced else "random"}.png')
    plt.show()

plot_bias_scores(random_sources_biases, random_response_biases, False)
plot_bias_scores(selected_sources_biases, selected_response_biases, True)