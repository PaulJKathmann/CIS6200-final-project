import json
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the bias_scores_4.json file
with open('./experiment/experiment_results/bias_scores_4.json', 'r') as file:
    data = json.load(file)

# Extract total_source_bias_score and output_bias_score for unfiltered data
sources_biases = []
response_biases = []

for key, value in data.items():
    if "output_bias_score" in value and "total_source_bias_score" in value:
        sources_biases.append(value["total_source_bias_score"])
        response_biases.append(value["output_bias_score"]["score"])

def filter_bias_scores(sources_biases, response_biases):
    """Filter out results with absolute bias output score < 0.05"""
    filtered_sources_biases = []
    filtered_response_biases = []
    for i in range(len(sources_biases)):
        if abs(response_biases[i]) >= 0.05:
            filtered_sources_biases.append(sources_biases[i])
            filtered_response_biases.append(response_biases[i])
    return filtered_sources_biases, filtered_response_biases

def plot_bias_scores(sources_biases, response_biases, filter: bool):
    if filter:
        sources_biases, response_biases = filter_bias_scores(sources_biases, response_biases)
        
    slope, intercept, r_value, _, _ = linregress(sources_biases, response_biases)
    regression_line = [slope * x + intercept for x in sources_biases]
    absolute_bias_score = sum([abs(x) for x in response_biases]) / len(response_biases)

    # Plot total_source_bias_score vs output_bias_score
    plt.figure(figsize=(10, 6))
    plt.scatter(sources_biases, response_biases, color='blue', alpha=0.7, label='Data Points')
    plt.plot(sources_biases, regression_line, color='red', label=f'Correlation Line (r={r_value:.2f})')
    # Add title, labels, and lege
    title = "Total Source Bias Score vs Output Bias Score"
    title_suffix = " (Filtered for Output Bias >= 0.05)" if filter else " (Unfiltered)"
    plt.title("Total Source Bias Score vs Output Bias Score" + title_suffix, fontsize=14)
    plt.xlabel("Total Source Bias Score", fontsize=12)
    plt.ylabel("Output Bias Score", fontsize=12)
    plt.legend(fontsize=10, title=f"Absolute Avg. Bias Score: {absolute_bias_score:.2f}")

    # Add grid and save the plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(f'/Users/paulkathmann/code/UPenn/CIS6200/CIS6200-final-project/experiment/experiment_results/bias_scores_plot_{"filtered" if filter else "unfiltered"}.png')
    plt.show()
    

plot_bias_scores(sources_biases, response_biases, False)
plot_bias_scores(sources_biases, response_biases, True)