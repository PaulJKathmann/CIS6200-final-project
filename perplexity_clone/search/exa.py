from typing import Optional
from pydantic_ai import Agent, RunContext
# from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tools

import requests
import os
from exa_py import Exa
from dotenv import load_dotenv


class MyExa:
    def __init__(self,
        api_key: Optional[str],
        base_url: str = "https://api.exa.ai",
        user_agent: str = "exa-py 1.9.1",
        ):
        """Initialize the Exa client with the provided API key and optional base URL.

        Args:
            api_key (str): The API key for authenticating with the Exa API.
            base_url (str, optional): The base URL for the Exa API. Defaults to "https://api.exa.ai".
        """
        if api_key is None:
            import os

            api_key = os.environ.get("EXA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided as an argument or in EXA_API_KEY environment variable"
                )
        self.base_url = base_url
        self.headers = {"x-api-key": api_key, "User-Agent": user_agent}
        
    def search(self, query: str, n: int = 3, params: Optional[dict] = None) -> list[str]:
        """Generate search queries for the given prompt.

        Args:
            query (str): The prompt to generate search queries for.
            n (int, optional): The number of search queries to generate. Defaults to 3.
            
        Returns:
            list[str]: A list of search queries.
        """
        url = f"{self.base_url}/search"
        params = {"query": query, "n": n}
        if params is not None:
            params.update(params)
        response = requests.post(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_contents(sefl, url: str) -> str:
        """Retrieve the contents of the web page at the given URL.

        Args:
            url (str): The URL of the web page to retrieve.

        Returns:
            str: The contents of the web page.
        """
        response = requests.get(url)
        response.raise_for_status()
        return response.text