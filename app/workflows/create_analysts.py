""""""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.models.models import Analyst, Perspectives, GenerateAnalystsState
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from app.config import settings
from app.prompts.prompts import analyst_instructions

class CreateAnalysts:
    def __init__(self):
        self.llm = settings.llm
        self.analyst_instructions = analyst_instructions


    def create_analysts(self, state: GenerateAnalystsState):
        
        """ Create analysts """
        
        topic=state['topic']
        max_analysts=state['max_analysts']
        human_analyst_feedback=state.get('human_analyst_feedback', '')
            
        # Enforce structured output
        structured_llm = self.llm.with_structured_output(Perspectives)

        # System message
        system_message = analyst_instructions.format(topic=topic,
                                                                human_analyst_feedback=human_analyst_feedback, 
                                                                max_analysts=max_analysts)

        # Generate question 
        analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
        
        # Write the list of analysis to state
        return {"analysts": analysts.analysts}

    @staticmethod
    def human_feedback(state: GenerateAnalystsState):
        pass

    @staticmethod
    def should_continue(state: GenerateAnalystsState):
        human_analyst_feedback = state.get('human_analyst_feedback', None)
        if human_analyst_feedback:
            return "create_analysts"
        return END

