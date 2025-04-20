import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel

import requests
import os
from exa_py import Exa
from exa_py.api import SearchResponse
from dotenv import load_dotenv

from perplexity_clone.BiasBert.biasbert import BiasBert
from perplexity_clone.utils import BiasLogger
from perplexity_clone.search.search import Search

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv('OPENAI_API_KEY')
exa = Exa(os.getenv('EXA_API_KEY'))
search_engine = Search(engine='exa')
bias_classifier = BiasBert()
bias_logger = BiasLogger()

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

bedrock_model = BedrockConverseModel('amazon.nova-lite-v1:0')

agent = Agent(model=bedrock_model,
            system_prompt='You are a helpful AI assistant that can answer questions by searching the web for information and then answering the question based on the information found. Do not change the users search prompt!',
            result_type=str,
        )



search_query_generator = Agent(
    model=bedrock_model,
    system_prompt='You are a helpful AI assistant that can answer questions, provide recommendations, and assist with a variety of tasks.',
    result_type=list[str]
)

summary_generator = Agent(
    model=bedrock_model,
    system_prompt='You are a helpful AI assistant that summarizes the given text so that it is useful to answer the given user prompt.',
    result_type=str
)

async def generate_search_queries(prompt, n = 3):
    print(f"Generating search queries for: {prompt}")
    
    user_prompt = f"""Break down the given user prompt into a series of search {n} queries that would be useful to find relevant information on this question {prompt}. 
    Each query should be a concise and specific search term or phrase that can be used to retrieve relevant information. 
    These queries can be in various formats, from simple keywords to more complex phrases. Do not add any formatting or numbering to the queries."""

    queries = await search_query_generator.run(user_prompt)
    print(queries)
    return queries

def calculate_bias_scores(bias_probabilities: list[list[float]]) -> list[float]:
    scores = []
    
    for bias in bias_probabilities:
        score = bias[0] * -1 + bias[2]
        scores.append(score)
    return scores


@agent.tool_plain
async def search(prompt: str, links_per_query: int = 2) -> list[dict[str, str]]:
    """
    Perform a web search using generated queries and summarize the results.


    Args:
        ctx (RunContext[str]): context containing the user prompt.
        links_per_query (int, optional): Number of returned links per search query. Defaults to 2.

    Returns:
        list[dict[str, str]]: List of search results where each result is a dictionary of title, URL, and summary.
    """
    search_queries = await generate_search_queries(prompt)
    results: list[dict[str, str]] = []
    total_bias_score = 0

    for i, query in enumerate(search_queries.data):
        results = search_engine.search_and_get_contents(query, links_per_query=links_per_query, highlights_only=True)
        text_results = [result['highlights'] for result in results]
        bias_probabilities = bias_classifier.classify_batch(text_results)
        print(f"Returned Bias probablities for searches {i=}: {bias_probabilities}")
        bias_scores = calculate_bias_scores(bias_probabilities)
        print(f"Returned Bias results {i=}: {bias_scores}")
        for i in range(len(results)):
            results[i]['bias'] = bias_scores[i]
            bias_logger.log_source_bias_score(prompt, bias_probabilities[i], bias_scores[i])
        results.extend(results)
    
        total_bias_score += sum(bias_scores) / len(bias_scores)
        # if -0.3 <= bias_score <= 0.3 or len(results) >= 10:
        #     break
    bias_logger.log_total_source_bias_score(prompt, total_bias_score / len(search_queries.data))
    print(f"Search tool done and returning: {results}")
    return results


def main():
    # Load prompts from data/prompts.csv and iterate over each row

    while True:
        query = input("Enter a query: ")
        result = agent.run_sync(query)
        print(f"Result: {result}")
        response_bias_probalities = bias_classifier.classify(result.data)
        response_bias = calculate_bias_scores([response_bias_probalities])
        bias_logger.log_output_bias_score(query, response_bias_probalities, response_bias[0], result.data)
        print(result.data)



if __name__ == "__main__":
    main()
    # run_experiment()