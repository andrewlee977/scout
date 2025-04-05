""""""

from app.workflows.create_analysts import *
from app.workflows.interview import *
from app.models.models import *
from langgraph.constants import Send
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from app.config import settings
from app.prompts.prompts import intro_conclusion_instructions, report_writer_instructions, podcast_prompt
from app.utils.llm_utils import invoke_llm


class ConductResearch:
    """
    A class that manages the research and report generation process.
    
    This class orchestrates the entire research workflow, from initiating interviews with analysts
    to generating a final report and podcast script. It coordinates the work of the InterviewBuilder
    class to gather information from various sources and synthesize it into coherent reports.
    
    The research process follows these steps:
    1. Initiate interviews with all analysts for a given topic
    2. Collect and process information from web, Wikipedia, and news sources
    3. Generate a comprehensive report based on the gathered information
    4. Create an introduction and conclusion for the report
    5. Finalize the report and generate a podcast script
    
    Attributes:
        llm: The language model used for generating reports, introductions, conclusions, and podcast scripts
        interview_builder: An instance of InterviewBuilder for managing analyst interviews
        podcast_prompt: The prompt template for generating podcast scripts
    
    Methods:
        initiate_all_interviews: Initiates interviews with all analysts or reroutes to analyst recreation
        write_report: Generates a comprehensive report based on the gathered information
        write_introduction: Creates an introduction for the final report
        write_conclusion: Creates a conclusion for the final report
        finalize_report: Combines the report with introduction and conclusion, and generates a podcast script
    """
    def __init__(self):
        self.llm = settings.llm
        self.interview_builder = InterviewBuilder()
        self.podcast_prompt = podcast_prompt
    
    @staticmethod
    def initiate_all_interviews(state: ResearchGraphState):
        """ Initiates all analyst interviews or reroutes to analyst recreation """
        human_analyst_feedback = state.get('human_analyst_feedback', 'approve')
        
        if human_analyst_feedback is None or human_analyst_feedback.lower() == 'approve':
            topic = state["topic"]
            return [Send("conduct_interview", {
                "analyst": analyst,
                "messages": [HumanMessage(
                    content=f"So you said you were writing an article on {topic}?"
                )]
            }) for analyst in state["analysts"]]
        
        return "create_analysts"

    
    def write_report(self, state: ResearchGraphState):
        """ Writes a report """
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
        report = invoke_llm(
            self.llm, 
            [SystemMessage(content=system_message)] + [HumanMessage(content=f"Write a report based upon these memos.")],
            function_name="write_report"
        )
        return {"content": report.content}


    def write_introduction(self, state: ResearchGraphState):
        """ Writes an introduction for final report"""
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        intro = invoke_llm(
            self.llm, 
            [SystemMessage(content=instructions)] + [HumanMessage(content=f"Write the report introduction")],
            function_name="write_introduction"
        )
        return {"introduction": intro.content}

    def write_conclusion(self, state: ResearchGraphState):
        """ Writes a conclusion for final report """
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        conclusion = invoke_llm(
            self.llm, 
            [SystemMessage(content=instructions)] + [HumanMessage(content=f"Write the report conclusion")],
            function_name="write_conclusion"
        )
        return {"conclusion": conclusion.content}


    def finalize_report(self, state: ResearchGraphState):
        """ Generate both formal report and podcast discussion """
        content = state['content']
        topic = state['topic']
        analysts = state['analysts']  # We'll need the analysts for the podcast version
        introduction = state.get('introduction', '')
        conclusion = state.get('conclusion', '')
        
        final_report = f"{introduction}\n\n{content}\n\n{conclusion}"
        formatted_analysts = [f"{a.name} ({a.role})" for a in analysts]
        
        system_message = self.podcast_prompt.format(topic=topic, analysts=formatted_analysts, content=content)
        
        podcast_version = invoke_llm(
            self.llm, 
            [SystemMessage(content=system_message)],
            function_name="generate_podcast"
        ).content
        
        return {
            "final_report": final_report,
            "podcast_script": podcast_version
        }