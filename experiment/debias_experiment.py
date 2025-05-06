"""
1. Loop over the filtered prompts and their bias scores.
2. For each prompt:
    2.1. Create the sub queries
    2.2. Perform the search
    2.3. Return the search results
    2.4. Log the bias scores of the sources
    2.5. Sort the results based on bias scores
    2.6. Run the algorithm to select a balanced set of sources 
    2.7. Compare LLM output for balanced and unbalanced sources set
        2.7.1 Ask the LLM the query and feed in the balanced sources -> log the bias scores
        2.7.2 Ask the LLM the query and feed in the unbalanced sources -> log the bias scores
    2.8. Save the results to a JSON file
 
"""
import asyncio
import json
import random
from typing import Union

from pydantic_ai import Agent
from perplexity_clone.BiasBert.biasbert import BiasBert
from perplexity_clone.agent.agent import calculate_bias_scores
from perplexity_clone.utils import BiasLogger
from perplexity_clone.agent.search_chaining import search, bias_classifier
from perplexity_clone.agent.search_chaining import bedrock_model

bias_classifier = BiasBert()
bias_logger = BiasLogger(log_file_path='experiment/experiment_results/debias_experiment.json', verbose=True)

agent = Agent(model=bedrock_model,
            system_prompt='You are a helpful AI assistant that can answer questions using the results it found online',
            result_type=str,
        )

def save_bias_scores(bias_scores: dict[str, tuple[float, float]], random_bias_scores: dict[str, tuple[float, float]] , file_path: str) -> None:
    """
    Save the bias scores to a JSON file.
    Args:
        bias_scores (dict[str, tuple[float, float]]): A dictionary containing the prompts and their corresponding bias scores.
        random_bias_scores (dict[str, tuple[float, float]]): A dictionary containing the prompts and their corresponding bias scores for the randomly choosen sources.
        file_path (str): The path to the JSON file where the bias scores will be saved.
    """
    merged_bias_scores = {}
    for prompt in bias_scores:
        merged_bias_scores[prompt] = {
            "balanced_source_bias_score": bias_scores[prompt][0],
            "balanced_response_bias_score": bias_scores[prompt][1],
            "unbalanced_source_bias_score": random_bias_scores[prompt][0],
            "unbalanced_response_bias_score": random_bias_scores[prompt][1]
        }
    try:
        with open(file_path, 'w') as f:
            json.dump(merged_bias_scores, f, indent=4)
        print(f"Bias scores saved to {file_path}")
    except Exception as e:
        print(f"Error saving bias scores: {e}")

def plot_bias_scores(bias_scores: dict[str, tuple[float, float]], unbalanced_bias_scores: dict[str, tuple[float, float]]) -> None:
    """
    Plot the bias scores for the experiment. The source bias scores are on the x-axis and the response bias scores are on the y-axis.
    The plot is saved as a PNG file in the experiment/experiment_results directory.
    Args:
        bias_scores (dict[str, tuple[float, float]]): A dictionary containing the prompts and their corresponding bias scores.
    """
    import matplotlib.pyplot as plt

    # Unzip the bias scores into two lists
    source_bias_scores, response_bias_scores = zip(*bias_scores.values())
    unbalanced_source_bias_scores, unbalanced_response_bias_scores = zip(*unbalanced_bias_scores.values())

    # Create a scatter plot for balanced results
    plt.figure(figsize=(10, 6))
    plt.scatter(source_bias_scores, response_bias_scores, alpha=0.5, label='Balanced')
    plt.title('Bias Scores Comparison (Balanced)')
    plt.xlabel('Source Bias Score')
    plt.ylabel('Response Bias Score')
    plt.grid()
    plt.legend()
    plt.savefig('experiment/experiment_results/bias_scores_plot_balanced.png')
    plt.show()

    # Create a scatter plot for unbalanced results
    plt.figure(figsize=(10, 6))
    plt.scatter(unbalanced_source_bias_scores, unbalanced_response_bias_scores, alpha=0.5, color='red', label='Unbalanced')
    plt.title('Bias Scores Comparison (Unbalanced)')
    plt.xlabel('Source Bias Score')
    plt.ylabel('Response Bias Score')
    plt.grid()
    plt.legend()
    plt.savefig('experiment/experiment_results/bias_scores_plot_unbalanced.png')
    plt.show()
    
    

def select_balanced_sources(prompt: str, sources_and_bias_tuples: list[tuple[str, float]], target_count=5, bias_threshold=0.15) -> tuple[list[str], float]:
    liberal_sources = []
    conservative_sources = []
    for source, bias_score in sources_and_bias_tuples:
        if bias_score < 0:
            liberal_sources.append((source, bias_score))
        else:
            conservative_sources.append((source, bias_score))
    
    liberal_sources.sort(key=lambda x: x[1])  # Sort by bias_score
    conservative_sources.sort(key=lambda x: x[1])  # Sort by bias_score
    selected_sources: list[str] = []
    total_source_bias = 0
    l, r = len(liberal_sources) - 1, 0

    # add the least liberal and the least conservative source first
    selected_sources.append(liberal_sources[l][0])
    selected_sources.append(conservative_sources[r][0])
    total_source_bias += liberal_sources[l][1] + conservative_sources[r][1]

    while len(selected_sources) < target_count or abs(total_source_bias) > bias_threshold:
        if total_source_bias <= 0:
            r += 1
            if r < len(conservative_sources):
                selected_sources.append(conservative_sources[r][0])
                total_source_bias += conservative_sources[r][1]
            else:
                break
        else:
            l -= 1
            if l >= 0:
                selected_sources.append(liberal_sources[l][0])
                total_source_bias += liberal_sources[l][1]
            else:
                break

    return selected_sources, total_source_bias

def select_random_sources(sources_and_bias_tuples: list[tuple[str, float]], n: int) -> tuple[list[str], float]:
    """Choose n random sources and calculate their total aggregate bias score.
    This function is used for the baseline comparison in the debias experiment.
    """
    selected_sources = random.sample(sources_and_bias_tuples, n)
    total_source_bias = sum([source[1] for source in selected_sources])
    return [source[0] for source in selected_sources], total_source_bias

async def debias_experiment():
    # load the prompts from the filtered bias scores
    try:
        with open('experiment/experiment_results/filtered_bias_scores.json', 'r') as f:
            data = json.load(f)
            prompts = list(data.keys())
    except FileNotFoundError:
        print("Filtered bias scores file not found. Please run the filter script first.")
        return
    except json.JSONDecodeError:
        print("Error decoding JSON from the filtered bias scores file.")
        return
    
    print(f"Loaded {len(prompts)} prompts from filtered bias scores file.")
    balanced_results: dict[str, tuple[float, float]] = {} # {prompt: (source_bias_score, response_bias_score)}
    unbalanced_results: dict[str, tuple[float, float]] = {} # {prompt: (source_bias_score, response_bias_score)}
    
    for prompt in prompts:
        print(f"Running debias experiment for prompt: {prompt}")
        search_results = await search(prompt, links_per_query=3)
        # Log the selected balanced sources
        
        source_texts = [source['highlights'] for source in search_results]
        biases_probabilities = bias_classifier.classify_batch(source_texts)
        bias_scores = calculate_bias_scores(biases_probabilities)
        sources_and_bias_tuples = [(source, bias) for source, bias in zip(source_texts, bias_scores)]
        
        selected_balanced_sources, total_source_bias_of_selected_sources = select_balanced_sources(prompt, sources_and_bias_tuples)
        selected_random_sources, total_source_bias_of_random_sources = select_random_sources(sources_and_bias_tuples, len(selected_balanced_sources))
        
        concat_prompt_balanced = prompt + "Search Results: " + " ".join(selected_balanced_sources)
        balanced_result = await agent.run(concat_prompt_balanced) 
        response_bias_probabilities = bias_classifier.classify(balanced_result.data)
        response_bias = calculate_bias_scores([response_bias_probabilities])
        balanced_results[prompt] = (total_source_bias_of_selected_sources, response_bias[0])
        
        concat_prompt_random = prompt + "Search Results: " + " ".join(selected_random_sources)
        random_result = await agent.run(concat_prompt_random) 
        response_bias_probabilities = bias_classifier.classify(random_result.data)
        response_bias_random = calculate_bias_scores([response_bias_probabilities])
        unbalanced_results[prompt] = (total_source_bias_of_random_sources, response_bias_random[0])
    
        
        
        
    
    plot_bias_scores(balanced_results, unbalanced_results)
    save_bias_scores(balanced_results, unbalanced_results, 'experiment/experiment_results/debias_experiment_results.json')

        
        
        
        
        
        
if __name__ == "__main__":
    # Run the debias experiment
    asyncio.run(debias_experiment())
    print("Debias experiment completed.")

        
    
    

    
    