from typing import Dict
from .agent import Agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class OrchestratorAgent(Agent):
    def __init__(self, tools):
        self.sys_msg = SystemMessage(content="""You are an orchestrator that asks clarifying questions when needed.
        Ask follow-up questions when:
        - The user's request is too vague
        - Time periods aren't specified
        - The type of news (tech, business, etc.) isn't clear
        - The desired level of detail isn't specified
        
        Keep questions concise and specific.""")
        
        self.current_context = {
            "needs_clarification": False,
            "pending_question": None,
            "gathered_info": ""
        }
        
        super().__init__(tools)

    async def process_message(self, user_input: str) -> Dict:
        """
        Process the message and determine if clarification is needed
        """
        if self.current_context["needs_clarification"]:
            # Handle response to our clarifying question
            complete_query = f"{self.current_context['gathered_info']} {user_input}"
            self.current_context["needs_clarification"] = False
            # Use parent class's process_message to get news
            response = super().process_message(complete_query)
            return {
                "response": response,
                "requires_follow_up": False
            }

        # Analyze if the initial request needs clarification
        analysis = await self._analyze_request(user_input)
        
        if analysis["needs_clarification"]:
            self.current_context.update({
                "needs_clarification": True,
                "pending_question": analysis["clarifying_question"],
                "gathered_info": user_input
            })
            return {
                "response": analysis["clarifying_question"],
                "requires_follow_up": True
            }
        
        # If no clarification needed, process normally
        response = super().process_message(user_input)
        return {
            "response": response,
            "requires_follow_up": False
        }

    async def _analyze_request(self, user_input: str) -> Dict:
        """
        Analyze if the request needs clarification
        """
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        
        analysis_prompt = f"""
        Analyze this news request: "{user_input}"
        
        If the request is clear and specific enough to fetch relevant news, respond with:
        CLEAR: Proceed with request
        
        If the request needs clarification, respond with:
        NEEDS_CLARIFICATION: <your single most important follow-up question>
        
        Examples:
        Request: "Tell me about news"
        Response: NEEDS_CLARIFICATION: What type of news are you interested in - technology, business, or general news?
        
        Request: "Tell me about the latest AI developments in tech"
        Response: CLEAR: Proceed with request
        """
        
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant that analyzes user requests."),
            HumanMessage(content=analysis_prompt)
        ])
        
        if "NEEDS_CLARIFICATION:" in response.content:
            question = response.content.split("NEEDS_CLARIFICATION:")[1].strip()
            return {
                "needs_clarification": True,
                "clarifying_question": question
            }
        
        return {
            "needs_clarification": False,
            "clarifying_question": None
        } 