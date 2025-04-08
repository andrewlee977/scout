from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from app.models.models import InterviewState, SearchQuery
from app.prompts.prompts import question_instructions, search_instructions, answer_instructions, section_writer_instructions
from app.config import settings
from app.utils.llm_utils import invoke_llm
from newsapi import NewsApiClient
import datetime
import logging

logger = logging.getLogger(__name__)

class InterviewBuilder:
    """
    A class that manages the interview process between analysts and an LLM.
    
    This class handles the generation of questions, searching for information from various sources
    (Tavily (Web), Wikipedia, NewsAPI), generating answers, and writing sections based on the gathered information.
    
    The interview process follows these steps:
    1. Generate a question based on the analyst's persona
    2. Search for relevant information from web, Wikipedia, and news sources
    3. Generate an answer based on the gathered information
    4. Write a section summarizing the analyst's insights
    
    Attributes:
        llm: The language model used for generating questions, answers, and sections
        question_instructions: Instructions for generating questions
        search_instructions: Instructions for searching for information
        answer_instructions: Instructions for generating answers
        section_writer_instructions: Instructions for writing sections
        todays_date: The current date in YYYY-MM-DD format
        tavily_search: Client for searching the web using Tavily
        news_api: Client for searching news using NewsAPI
    
    Methods:
        generate_question: Generates a question based on the analyst's persona
        search_web: Searches the web for information using Tavily
        search_wikipedia: Searches Wikipedia for information
        search_news: Searches news articles using NewsAPI
        generate_answer: Generates an answer based on the gathered information
        save_interview: Saves the interview transcript
        route_messages: Routes between question and answer based on the interview state
        write_section: Writes a section summarizing the analyst's insights
    """
    def __init__(self):
        # Initialize LLM
        self.llm = settings.llm

        # Initialize instructions
        self.question_instructions = question_instructions
        self.search_instructions = search_instructions
        self.answer_instructions = answer_instructions
        self.section_writer_instructions = section_writer_instructions
        self.todays_date = datetime.date.today().strftime("%Y-%m-%d")
        
        # Initialize search clients
        self.tavily_search = TavilySearchResults(
            api_key=settings.TAVILY_API_KEY,
            max_results=5
        )
        self.news_api = NewsApiClient(api_key=settings.NEWS_API_KEY)

    def generate_question(self, state: InterviewState):
        """ Node to generate a question """
        analyst = state["analyst"]
        messages = state["messages"]
        system_message = self.question_instructions.format(goals=analyst.persona)
        question = invoke_llm(self.llm, [SystemMessage(content=system_message)] + messages, function_name="generate_question")
        return {"messages": [question]}

    def search_web(self, state: InterviewState):
        """ Retrieve docs from web search """
        try:
            # Initialize Tavily client
            logger.info("Initializing Tavily client...")
            if not settings.TAVILY_API_KEY:
                raise ValueError("TAVILY_API_KEY is not set")

            system_message = self.search_instructions.format(todays_date=self.todays_date)
            structured_llm = self.llm.with_structured_output(SearchQuery)
            search_query = invoke_llm(
                structured_llm, 
                [SystemMessage(content=system_message)] + state['messages'],
                function_name="search_web"
            )
            search_docs = self.tavily_search.invoke(search_query.search_query)
            
            # Handle both string and dictionary formats
            formatted_docs = []
            for doc in search_docs:
                if isinstance(doc, dict):
                    # Handle dictionary format
                    url = doc.get('url', '')
                    content = doc.get('content', '')
                    formatted_docs.append(f'<Document href="{url}"/>\n{content}\n</Document>')
                else:
                    # Handle string format
                    formatted_docs.append(f'<Document>\n{doc}\n</Document>')
                
                formatted_search_docs = "\n\n---\n\n".join(formatted_docs)
                return {"context": [formatted_search_docs]}
        except Exception as e:
            logger.error(f"Error in search_web: {str(e)}", exc_info=True)
            return {"context": ["Error retrieving web search results. Please try again later."]}

    def search_wikipedia(self, state: InterviewState):
        """ Retrieve docs from wikipedia """
        try:
            structured_llm = self.llm.with_structured_output(SearchQuery)
            system_message = self.search_instructions.format(todays_date=self.todays_date)
            search_query = invoke_llm(
                structured_llm, 
                [SystemMessage(content=system_message)] + state['messages'],
                function_name="search_wikipedia"
            )
            
            # Create the loader here with the query
            wiki_loader = WikipediaLoader(query=search_query.search_query, load_max_docs=2)
            search_docs = wiki_loader.load()
            
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                    for doc in search_docs
                ]
            )
            return {"context": [formatted_search_docs]}
        except Exception as e:
            logger.error(f"Error in search_wikipedia: {str(e)}", exc_info=True)
            return {"context": ["Error retrieving wikipedia search results. Please try again later."]}
    
    def search_news(self, state: InterviewState):
        """ Retrieve docs from NewsAPI"""
        try:
            # Initialize NewsAPI client
            logger.info("Initializing NewsAPI client...")
            if not settings.NEWS_API_KEY:
                raise ValueError("NEWS_API_KEY is not set")

            structured_llm = self.llm.with_structured_output(SearchQuery)
            system_message = self.search_instructions.format(todays_date=self.todays_date)
            search_query = invoke_llm(
                structured_llm, 
                [SystemMessage(content=system_message)] + state['messages'],
                function_name="search_news"
            )
            
            logger.info(f"Making NewsAPI request with query: {search_query.search_query}")
            
            # Add more parameters to help with debugging
            search_params = {
                'q': search_query.search_query,
                'language': 'en',
                'sort_by': 'relevancy',
                'page': 1,
                'page_size': 10
            }
            logger.info(f"NewsAPI request parameters: {search_params}")
            
            search_docs = self.news_api.get_everything(**search_params)
            
            if not search_docs:
                logger.warning("Empty response from NewsAPI")
                return {"context": ["No recent news articles found."]}
                
            if 'articles' not in search_docs:
                logger.warning(f"Unexpected NewsAPI response structure: {search_docs}")
                return {"context": ["Error in news API response format."]}
            
            if not search_docs['articles']:
                logger.warning("No articles found in NewsAPI response")
                return {"context": ["No recent news articles found."]}
            
            formatted_search_docs = [f'<Document source="{doc["url"]}" published={doc["publishedAt"]}/>\n{doc["content"]}\n</Document>' for doc in search_docs["articles"]]
            return {"context": [formatted_search_docs]}
            
        except Exception as e:
            logger.error(f"Error in search_news: {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return {"context": ["Error retrieving news articles. Please try again later."]}

    def generate_answer(self, state: InterviewState):
        """ Node to answer a question """
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]
        system_message = answer_instructions.format(goals=analyst.persona, context=context)
        answer = invoke_llm(self.llm, [SystemMessage(content=system_message)] + messages, function_name="generate_answer")
        answer.name = "expert"
        return {"messages": [answer]}

    @staticmethod
    def route_messages(state: InterviewState, name: str = "expert"):
        """ Route between question and answer """
        messages = state["messages"]
        max_num_turns = state.get('max_num_turns', 2)
        num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
        if num_responses >= max_num_turns:
            return 'save_interview'
        last_question = messages[-2]
        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'
        return "ask_question"
    
    @staticmethod
    def save_interview(state: InterviewState):
        """ Save interviews """
        messages = state["messages"]
        interview = get_buffer_string(messages)
        return {"interview": interview}
    
    def write_section(self, state: InterviewState):
        """ Node to answer a question """
        context = state["context"]
        analyst = state["analyst"]
        system_message = section_writer_instructions.format(focus=analyst.description)
        section = invoke_llm(
            self.llm, 
            [SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {context}")],
            function_name="write_section"
        )
        return {"sections": [section.content]}
