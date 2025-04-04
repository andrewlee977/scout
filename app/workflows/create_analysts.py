""""""

from langchain_core.messages import SystemMessage, HumanMessage
from app.models.models import Perspectives, GenerateAnalystsState
from langgraph.graph import END
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
        num_themes = max_analysts - 1
            
        # Enforce structured output
        structured_llm = self.llm.with_structured_output(Perspectives)

        # System message
        system_message = analyst_instructions.format(topic=topic,
                                                    human_analyst_feedback=human_analyst_feedback, 
                                                    num_themes=num_themes,
                                                    max_analysts=max_analysts)

        # Generate question 
        analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
        
        # Write the list of analysis to state
        return {"analysts": analysts.analysts}

    @staticmethod
    def human_feedback(state: GenerateAnalystsState):
        pass
