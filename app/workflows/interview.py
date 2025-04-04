from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from app.models.models import InterviewState, SearchQuery
from app.prompts.prompts import question_instructions, search_instructions, answer_instructions, section_writer_instructions
from app.config import settings
from newsapi import NewsApiClient
import datetime

# Initialize Tavily search with API key from settings
tavily_search = TavilySearchResults(
    api_key=settings.TAVILY_API_KEY,
    max_results=3
)
news_api = NewsApiClient(api_key=settings.NEWS_API_KEY)

class InterviewBuilder:
    def __init__(self):
        self.llm = settings.llm
        self.question_instructions = question_instructions
        self.search_instructions = search_instructions
        self.answer_instructions = answer_instructions
        self.section_writer_instructions = section_writer_instructions
        self.todays_date = datetime.date.today().strftime("%Y-%m-%d")

    def generate_question(self, state: InterviewState):
        """ Node to generate a question """
        analyst = state["analyst"]
        messages = state["messages"]
        system_message = self.question_instructions.format(goals=analyst.persona)
        question = self.llm.invoke([SystemMessage(content=system_message)] + messages)
        return {"messages": [question]}

    def search_web(self, state: InterviewState):
        """ Retrieve docs from web search """
        system_message = self.search_instructions.format(todays_date=self.todays_date)
        structured_llm = self.llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([SystemMessage(content=system_message)] + state['messages'])
        search_docs = tavily_search.invoke(search_query.search_query)
        
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

    def search_wikipedia(self, state: InterviewState):
        """ Retrieve docs from wikipedia """
        structured_llm = self.llm.with_structured_output(SearchQuery)
        system_message = self.search_instructions.format(todays_date=self.todays_date)
        search_query = structured_llm.invoke([SystemMessage(content=system_message)] + state['messages'])
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        return {"context": [formatted_search_docs]}
    
    def search_news(self, state: InterviewState):
        """ Retrieve docs from NewsAPI"""
        structured_llm = self.llm.with_structured_output(SearchQuery)
        system_message = self.search_instructions.format(todays_date=self.todays_date)
        search_query = structured_llm.invoke([SystemMessage(content=system_message)] + state['messages'])
        
        search_docs = news_api.get_everything(q=search_query.search_query, language='en', sort_by='relevancy', page=1, page_size=10)
        print('SEARCH QUERY NEWS=================', search_query.search_query)

        formatted_search_docs = [f'<Document source="{doc["url"]}" published={doc["publishedAt"]}/>\n{doc["content"]}\n</Document>' for doc in search_docs["articles"]]
        print(f"Formatted search docs: {formatted_search_docs}")

        return {"context": [formatted_search_docs]}

    def generate_answer(self, state: InterviewState):
        """ Node to answer a question """
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]
        system_message = answer_instructions.format(goals=analyst.persona, context=context)
        answer = self.llm.invoke([SystemMessage(content=system_message)] + messages)
        answer.name = "expert"
        return {"messages": [answer]}

    @staticmethod
    def save_interview(state: InterviewState):
        """ Save interviews """
        messages = state["messages"]
        interview = get_buffer_string(messages)
        return {"interview": interview}
    

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


    def write_section(self, state: InterviewState):
        """ Node to answer a question """
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
        system_message = section_writer_instructions.format(focus=analyst.description)
        section = self.llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {context}")])
        return {"sections": [section.content]}
