"""Tools Registry"""

import requests
import logging
import os
from typing import List
from openai import OpenAI
from app.config import settings  # Use the existing instance, not the class




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
    # Adjust number of articles based on summary length
    # articles_count = {
    #     "short": 5,
    #     "medium": 10,
    #     "long": 15
    # }.get(summary_length, 5)
    
    # Test APIs first
    # await test_google_credentials()
    _test_openai_api()

    articles = _fetch_latest_news()  # Use adjusted count

    summary = _create_batch_summary(articles)

    
    return summary

    # return {
    #     "summary": summary,
    #     "audio_content": audio
    # } 




def get_all_tools():
    """Return the list of all registered tool functions."""
    return TOOLS_REGISTRY
