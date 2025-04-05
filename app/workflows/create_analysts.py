""""""

from langchain_core.messages import SystemMessage, HumanMessage
from app.models.models import Perspectives, GenerateAnalystsState
from app.config import settings
from app.prompts.prompts import analyst_instructions
from app.utils.llm_utils import invoke_llm

class CreateAnalysts:
    """
    A class that creates a team of AI analysts with diverse expertise and perspectives.
    
    This class is responsible for generating a set of AI analysts, each with a unique persona,
    expertise, and perspective. These analysts will be used to conduct interviews and provide
    diverse viewpoints on a given topic.
    
    The analyst creation process follows these steps:
    1. Generate a list of analyst personas with diverse backgrounds and expertise
    2. Create Analyst objects with names, descriptions, and personas
    3. Return the list of analysts for use in the interview process
    
    Attributes:
        llm: The language model used for generating analyst personas
        analyst_instructions: Instructions for generating analyst personas
    
    Methods:
        create_analysts: Generates a list of Analyst objects with diverse expertise
    """
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
        analysts = invoke_llm(
            structured_llm, 
            [SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")],
            function_name="create_analysts"
        )

        # Write the list of analysis to state
        return {"analysts": analysts.analysts}

    @staticmethod
    def human_feedback(state: GenerateAnalystsState):
        pass
