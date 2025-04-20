import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
import requests
import os
from exa_py import Exa
from exa_py.api import SearchResponse

load_dotenv()
exa = Exa(os.getenv('EXA_API_KEY'))

summary_generator = Agent(
    'openai:gpt-4o-mini',
    system_prompt='You are a helpful AI assistant that summarizes the given text so that it is useful to answer the given user prompt.',
    result_type=str
)

class Search:
    def __init__(self, engine='exa'):
        self.engine = None
        if engine == 'exa':
            self.engine = Exa(os.getenv('EXA_API_KEY'))
        else:
            raise ValueError(f"Unsupported search engine: {engine}")
        

    def search_and_get_links(self, query, links_per_query=2):
        """
        Perform a search using the specified engine and return the results.

        Args:
            query (str): The search query.
            links_per_query (int): The number of links to retrieve per query.

        Returns:
            list: A list of search results.
        """
        if self.engine == 'exa':
            # Perform the search using Exa API
            search_response = exa.search(query, num_results=links_per_query)
            search_results = [result['title'] for result in search_response['results']]
            return search_results

        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
        
    def get_contents(self, url):
        """
        Fetch the contents of a given URL.

        Args:
            url (str): The URL to fetch.

        Returns:
            str: The contents of the URL.
        """
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError(f"Failed to fetch URL: {url}, Status code: {response.status_code}")
        
    def search_and_get_contents(self, query, links_per_query=2, highlights_only=True) -> list[dict[str, str]]:
        """
        Perform a search using the specified engine and return the results and their contents.

        Args:
            query (str): The search query.
            links_per_query (int): The number of links to retrieve per query.

        Returns:
            list: A list of dictionaries containing search results and their contents.
        """
        
        search_response = self.engine.search_and_contents(query,
            num_results=links_per_query,
            use_autoprompt=False,
            highlights=highlights_only
        )
        results = []
        for result in search_response.results:
            results.append({
                'title': result.title,
                'url': result.url,
                'highlights': "/n".join(result.highlights),
                'text': result.text
            })
        return results
    
    async def summarize_search_results(self, search_response: SearchResponse):
        text_results = [result.text for result in search_response.results]
        print(f"Summarizing {len(text_results)} results")
        print(f"Results: {text_results}")
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
    
    

           
