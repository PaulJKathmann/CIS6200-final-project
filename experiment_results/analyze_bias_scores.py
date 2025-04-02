import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Load the bias scores JSON file
with open('experiment_results/bias_scores.json', 'r') as file:
    data = json.load(file)

# Extract total_source_bias_score and output_bias_score for each element
prompts = []
total_source_bias_scores = []
output_bias_scores = []

for prompt, scores in data.items():
    prompts.append(prompt)
    total_source_bias_scores.append(scores["total_source_bias_score"])
    output_bias_scores.append(scores["output_bias_score"]["score"])

# Calculate correlation and regression line
slope, intercept, r_value, _, _ = linregress(total_source_bias_scores, output_bias_scores)
regression_line = [slope * x + intercept for x in total_source_bias_scores]

# Plot total_source_bias_score vs output_bias_score
plt.figure(figsize=(10, 6))
plt.scatter(total_source_bias_scores, output_bias_scores, color='blue', alpha=0.7, label='Data Points')
plt.plot(total_source_bias_scores, regression_line, color='red', label=f'Correlation Line (r={r_value:.2f})')

# Add title, labels, and legend
plt.title("Total Source Bias Score vs Output Bias Score", fontsize=14)
plt.xlabel("Total Source Bias Score", fontsize=12)
plt.ylabel("Output Bias Score", fontsize=12)
plt.legend(fontsize=10)

# Annotate points with their corresponding prompts
for i, prompt in enumerate(prompts):
    plt.annotate(prompt, (total_source_bias_scores[i], output_bias_scores[i]), fontsize=8, alpha=0.7)

# Add grid and save the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('experiment_results/bias_scores_scatter_plot.png')
plt.show()
