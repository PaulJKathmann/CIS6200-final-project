import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from botocore.exceptions import ClientError
import requests
import os
from exa_py import Exa
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
            system_prompt='You are a helpful AI assistant that can answer questions using the results it found online',
            result_type=str,
        )

search_query_generator = Agent(
    model='openai:gpt-4o-mini',
    system_prompt='You are a helpful AI assistant that can answer questions, provide recommendations, and assist with a variety of tasks.',
    result_type=list[str]
)

async def generate_search_queries(prompt, n = 3):
    print(f"Generating search queries for: {prompt}")
    
    user_prompt = f"""Break down the given user prompt into a series of search {n} queries that would be useful to find relevant information on this question {prompt}. 
    Each query should be a concise and specific search term or phrase that can be used to retrieve relevant information. 
    These queries can be in various formats, from simple keywords to more complex phrases. Do not add any formatting or numbering to the queries."""
    
    for attempt in range(1, 10):
        try:
            queries = await search_query_generator.run(user_prompt)
            print(queries)
            return queries
        except Exception as e:
            if e is not ClientError:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < 10:  # Retry only if it's not the last attempt
                    await asyncio.sleep((3**attempt) // 3)  # Exponential backoff
                else:
                    raise e


def calculate_bias_scores(bias_probabilities: list[list[float]]) -> list[float]:
    scores = []
    
    for bias in bias_probabilities:
        score = bias[0] * -1 + bias[2]
        scores.append(score)
    return scores

async def search(prompt: str, links_per_query: int = 4) -> list[dict[str, str]]:
    """
    Perform a web search using by generating queries, getting the contents for each query and return .


    Args:
        ctx (RunContext[str]): context containing the user prompt.
        links_per_query (int, optional): Number of returned links per search query. Defaults to 2.

    Returns:
        list[dict[str, str]]: List of search results where each result is a dictionary of title, URL, and summary.
    """
    search_queries = None
    search_queries = await generate_search_queries(prompt, n=5)
            
    results: list[dict[str, str]] = []

    for i, query in enumerate(search_queries.data):
        query_results = search_engine.search_and_get_contents(query, links_per_query=links_per_query, highlights_only=True)
        results.extend(query_results)
    return results

async def search_and_log_bias(prompt: str, links_per_query: int = 2) -> list[dict[str, str]]:
    """
    Perform a web search using by generating queries, getting the contents for each query and return it. 
    This function also logs the bias scores of the sources.


    Args:
        ctx (RunContext[str]): context containing the user prompt.
        links_per_query (int, optional): Number of returned links per search query. Defaults to 2.

    Returns:
        list[dict[str, str]]: List of search results where each result is a dictionary of title, URL, and summary.
    """
    search_queries = None
    for attempt in range(1, 6):
        try:
            search_queries = await generate_search_queries(prompt)
            break
        except Exception as e:
            if e is not ClientError:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < 5:  # Retry only if it's not the last attempt
                    await asyncio.sleep((3**attempt) // 3)  # Exponential backoff
                else:
                    raise e 
            
    results: list[dict[str, str]] = []
    total_bias_score = 0

    for i, query in enumerate(search_queries.data):
        query_results = search_engine.search_and_get_contents(query, links_per_query=links_per_query, highlights_only=True)
        text_results = [result['highlights'] for result in query_results]
        bias_probabilities = bias_classifier.classify_batch(text_results)
        print(f"Returned Bias probablities for searches {i=}: {bias_probabilities}")
        bias_scores = calculate_bias_scores(bias_probabilities)
        print(f"Returned Bias results {i=}: {bias_scores}")
        for i in range(len(query_results)):
            query_results[i]['bias'] = bias_scores[i]
            bias_logger.log_source_bias_score(prompt, bias_probabilities[i], bias_scores[i])

        results.extend(query_results)
    
        total_bias_score += sum(bias_scores) / len(bias_scores)
        # if -0.3 <= bias_score <= 0.3 or len(results) >= 10:
        #     break
    bias_logger.log_total_source_bias_score(prompt, total_bias_score / len(search_queries.data))
    print(f"Search tool done and returning: {results}")
    return results


async def main():
    # Load prompts from data/prompts.csv and iterate over each row

    while True:
        prompt = input("Enter a query: ")
        search_results = await search(prompt, links_per_query=2)
        print(f"Result: {search_results}")
        query = prompt + "Search Results: "
        for result in search_results:
            for key, value in result.items():
                query += f"{key}: {value}"
        result = await agent.run(query)
        
        
        response_bias_probalities = bias_classifier.classify(result.data)
        response_bias = calculate_bias_scores([response_bias_probalities])
        bias_logger.log_output_bias_score(prompt, response_bias_probalities, response_bias[0], result.data)
        print(result.data)


if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    asyncio.run(main())
    print("Bias scores saved to file.")
