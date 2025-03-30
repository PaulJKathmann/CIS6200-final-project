import asyncio
from pydantic_ai import Agent, RunContext
# from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tools

import requests
import os
from exa_py import Exa
from exa_py.api import SearchResponse
from dotenv import load_dotenv

from perplexity_clone.BiasBert.biasbert import BiasBert

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv('OPENAI_API_KEY')
exa = Exa(os.getenv('EXA_API_KEY'))
bias_classifier = BiasBert()

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")



agent = Agent(
    'openai:o1',
    system_prompt='You are a helpful AI assistant that can answer questions by searching the web for information and then answering the question based on the information found.',
    result_type=str
)

search_query_generator = Agent(
    'openai:o1',
    system_prompt='You are a helpful AI assistant that can answer questions, provide recommendations, and assist with a variety of tasks.',
    result_type=list[str]
)

summary_generator = Agent(
    'openai:gpt-4o-mini',
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

async def summarize_text(search_response: SearchResponse):
    text_results = [result.text for result in search_response.results]
    summarized_results = await asyncio.gather(
        *[
            summary_generator.run(result.text)
            for result in search_response.results
        ]
    )
    summarized_results = [
        {
            'title': result.title,
            'url': result.url,
            'summary': summary.data
        }
        for result, summary in zip(search_response.results, summarized_results)
    ]
    return summarized_results

@agent.tool
async def search(ctx: RunContext[str], links_per_query: int = 2) -> list[dict[str, str]]:
    search_queries = await generate_search_queries(ctx.prompt)
    results = []
    bias_score = 0

    #while True:
    for i, query in enumerate(search_queries.data):
        search_response = exa.search_and_contents(query,
            num_results=links_per_query,
            use_autoprompt=False,
            
        )
        summarized_results = await summarize_text(search_response)

        summary_texts = [result['summary'] for result in summarized_results]
        bias_probabilities = bias_classifier.classify_batch(summary_texts)
        print(f"Returned Bias probablities for searches {i=}: {bias_probabilities}")
        bias_scores = calculate_bias_scores(bias_probabilities)
        print(f"Returned Bias results {i=}: {bias_scores}")
        for i, result in enumerate(summarized_results):
            summarized_results[i]['bias'] = bias_scores[i]
            
        results.extend(summarized_results)
        bias_score += sum(bias_scores) / len(bias_scores)
        print(f"Current Bias Score: {bias_score}")
        # if -0.3 <= bias_score <= 0.3 or len(results) >= 10:
        #     break

    return results



def main():
    while True:
        query = input("Enter a query: ")
        result = agent.run_sync(query)
        response_bias_probalities = bias_classifier.classify(result.data)
        response_bias = calculate_bias_scores([response_bias_probalities])
        print(f"Response Bias: {response_bias[0]}")
        print(result.data)



if __name__ == "__main__":
    # result = agent.run_sync(
    #     'Can you list the top five highest-grossing animated films of 2025?'
    # )
    # print(result.data)
    main()