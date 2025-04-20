from asyncio import sleep
import asyncio
import os
from exa_py import Exa
from dotenv import load_dotenv
from pydantic_ai import Agent

from perplexity_clone.BiasBert.biasbert import BiasBert
from perplexity_clone.utils import BiasLogger
from pydantic_ai.models.bedrock import BedrockConverseModel
import csv
from perplexity_clone.agent.agent import calculate_bias_scores
from perplexity_clone.agent.search_chaining import search_and_log_bias, bedrock_model, bias_logger

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv('OPENAI_API_KEY')
exa = Exa(os.getenv('EXA_API_KEY'))
bias_classifier = BiasBert()

agent = Agent(model=bedrock_model,
            system_prompt='You are a helpful AI assistant that can answer questions using the results it found online',
            result_type=str,
        )


async def run_experiment(num_prompts: int = 5):
    if num_prompts < 1 or num_prompts > 100:
        raise ValueError("num_prompts must be between 1 and 100")
    prompts = []
    with open('./data/prompts.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row:  # Ensure the row is not empty
                prompts.append(row[0])
    

    print(f"Loaded {len(prompts)} prompts from data/prompts.csv")
    for query in prompts[len(prompts) - num_prompts:]:
        search_results = await search_and_log_bias(query, links_per_query=2)
        print(f"Result: {search_results}")
        concat_prompt = query + "Search Results: "
        for result in search_results:
            for key, value in result.items():
                concat_prompt += f"{key}: {value}"
        result = await agent.run(concat_prompt)
    
        response_bias_probalities = bias_classifier.classify(result.data)
        response_bias = calculate_bias_scores([response_bias_probalities])
        bias_logger.log_output_bias_score(query, response_bias_probalities, response_bias[0], result.data)
        print("\n\n\n")
        
        
if __name__ == "__main__":
    asyncio.run(run_experiment(num_prompts=50))
    print("Bias scores saved to file.")