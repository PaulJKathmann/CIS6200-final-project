import json

# Load the JSON data
input_file = 'experiment/experiment_results/bias_scores_4.json'
output_file = 'experiment/experiment_results/filtered_bias_scores.json'

with open(input_file, 'r') as f:
    data = json.load(f)

# Filter out results with absolute bias output score < 0.05
new_data = {}
for key, value in data.items():
    print(key)
    print(value)
    if 'output_bias_score' in value:
        value['output_bias_score'] = value['output_bias_score'].get('score', 0)
        if value['output_bias_score'] is None:
            value['output_bias_score'] = 0
    else:
        value['output_bias_score'] = 0
    
    if abs(value['output_bias_score']) >= 0.05:
        new_data[key] = value

# Save the filtered results to a new JSON file
with open(output_file, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Filtered results saved to {output_file}")