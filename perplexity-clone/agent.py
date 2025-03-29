from pydantic_ai import Agent, RunContext
# from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tools
import requests
import os
from exa_py import Exa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv('OPENAI_API_KEY')
exa = Exa(os.getenv('EXA_API_KEY'))

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")



agent = Agent(
    'openai:o1',
    system_prompt='You are a helpful AI assistant that can answer questions, provide recommendations, and assist with a variety of tasks. You can also search the web for information. How can I help you today?',
    result_type=str
)

search_query_generator = Agent(
    'openai:o1',
    system_prompt='You are a helpful AI assistant that can answer questions, provide recommendations, and assist with a variety of tasks.',
    result_type=list[str]
)

async def generate_search_queries(prompt, n = 3):
    user_prompt = f"""Break down the given user prompt into a series of search {n} queries that would be useful to find relevant information on this question {prompt}. 
    Each query should be a concise and specific search term or phrase that can be used to retrieve relevant information. 
    These queries can be in various formats, from simple keywords to more complex phrases. Do not add any formatting or numbering to the queries."""

    completion = await search_query_generator.run(
        user=user_promptWhich 
    )
    return [s.strip() for s in completion.split('\n') if s.strip()][:n]


@agent.tool
def search(ctx: RunContext[str], links_per_query: int = 2):
    search_queries = generate_search_queries(ctx.prompt)
    print(search_queries)
    results = []
    for query in search_queries:
        search_response = exa.search_and_contents(query,
            num_results=links_per_query,
            use_autoprompt=False
        )
        results.extend(search_response.results)
    print(results)
    return results



def main():
    while True:
        query = input("Enter a query: ")
        result = agent.run_sync(query)
        print(result.data)



if __name__ == "__main__":
    # result = agent.run_sync(
    #     'Can you list the top five highest-grossing animated films of 2025?'
    # )
    # print(result.data)
    main()