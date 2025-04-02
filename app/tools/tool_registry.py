"""Tools Registry"""

import requests
import logging
import inspect
from typing import List
from openai import OpenAI
from app.config import settings  # Use the existing instance, not the class
from functools import wraps

logger = logging.getLogger(__name__)

# Global registry for all tool functions
TOOLS_REGISTRY = []

def register_tool(func):
    """
    Decorator that registers a tool function.
    Every function decorated with @register_tool will be added to the TOOLS_REGISTRY.
    """
    TOOLS_REGISTRY.append(func)
    return func

# === Tool Definitions ===


def _fetch_latest_news() -> int:
    """Fetches the latest news articles"""
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": settings.NEWS_API_KEY,  # Use the global instance
            "category": "technology",
            "language": "en",
            "pageSize": 10
        }
        
        logger.info(f"Fetching news articles from {url}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json()["articles"]
        logger.info(f"Successfully fetched {len(articles)} articles")
        return articles
    except Exception as e:
        logger.error(f"Error fetching news articles: {str(e)}", exc_info=True)
        raise


def _create_batch_summary(articles: List[dict]) -> str:
    """Creates a summary of the news articles"""
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        articles_text = "\n\n".join([
            f"Title: {article['title']}\n"
            f"Content: {article['content'] or article['description']}"
            for article in articles
        ])
        
        # max_tokens = {
        #     "short": 150,
        #     "medium": 250,
        #     "long": 350
        # }.get(summary_length, 150)
        
        logger.info(f"Creating news summary")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
                Please provide a concise summary of today's tech news, based on these articles:
                
                {articles_text}
                
                Create a coherent, conversational summary that connects the main themes and developments.
                Keep it engaging.
                """
            }],
            # max_tokens=max_tokens,
            temperature=0.7,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        summary = response.choices[0].message.content
        logger.info("Successfully created summary")
        return summary
    except Exception as e:
        logger.error(f"Error creating summary: {str(e)}", exc_info=True)
        raise

def _test_openai_api():
    """Function test OpenAI API connection"""
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        logger.info("OpenAI API test successful")
        return True
    except Exception as e:
        logger.error(f"OpenAI API test failed: {str(e)}")
        return False




@register_tool
def get_news_briefing() -> dict:
    """Get a comprehensive briefing of the latest tech news.
    
    This is the main tool for getting tech news updates. It will:
    1. Fetch the most recent tech news articles
    2. Create a summary of the key developments
    3. Return a formatted briefing
    """

    _test_openai_api()

    articles = _fetch_latest_news()  # Use adjusted count

    summary = _create_batch_summary(articles)

    
    return summary




@register_tool
def search_web_context(user_query: str, topic: str, days: int, search_depth: str = "basic") -> str:
    """
    Performs an intelligent web search using Tavily API to provide contextual information.
    
    This tool complements the news briefing functionality by searching across the broader internet
    for relevant context, explanations, and background information. It can be used to:
    - Provide historical context for current news
    - Find detailed explanations of technical concepts
    - Gather multiple perspectives on a topic
    - Search for specific facts or statistics
    
    Args:
        user_query (str): The search query to be processed
        topic (str): The topic of the search
        days (int): The number of days to search
        search_depth (str, optional): Level of search depth. 
            "basic" - Quick search for general information
            "deep" - More comprehensive search including analysis and academic sources
            Defaults to "basic".

    Returns:
        str: Curated search results formatted as a coherent response
        
    Example:
        query = "What is the background of Silicon Valley Bank collapse?"
        result = search_web_context(query, topic, days, search_depth="deep")
    """

    url = "https://api.tavily.com/search"

    payload = {
        "query": user_query,
        "topic": topic,
        "search_depth": search_depth,
        "chunks_per_source": 3,
        "max_results": 5,
        "time_range": None,
        "days": days,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_domains": [],
        "exclude_domains": []
    }
    headers = {
        "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text


def get_all_tools():
    """Returns all available tools."""
    logger.info("Registering tools...")  # Add this to verify tools are being registered
    
    tools = TOOLS_REGISTRY    

    logger.info(f"Registered tools: {[tool.__name__ for tool in tools]}")
    return tools


def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
