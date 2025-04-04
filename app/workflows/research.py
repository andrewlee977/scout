""""""

from app.workflows.create_analysts import *
from app.workflows.interview import *
from app.models.models import *
from langgraph.constants import Send
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from app.config import settings
from app.prompts.prompts import intro_conclusion_instructions, report_writer_instructions


class ConductResearch:
    def __init__(self):
        self.llm = settings.llm
        self.interview_builder = InterviewBuilder()
    
    @staticmethod
    def initiate_all_interviews(state: ResearchGraphState):
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
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
        report = self.llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
        return {"content": report.content}


    def write_introduction(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        intro = self.llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
        return {"introduction": intro.content}

    def write_conclusion(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
            
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        conclusion = self.llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
        return {"conclusion": conclusion.content}


    def finalize_report(self, state: ResearchGraphState):
        """ Generate both formal report and podcast discussion """
        content = state['content']
        topic = state['topic']
        analysts = state['analysts']  # We'll need the analysts for the podcast version

        # Generate formal report (keep existing logic)
        introduction_prompt = f"""Write an introduction for a report on the following topic: {topic}"""
        conclusion_prompt = f"""Write a conclusion for a report on the following topic: {topic}"""

        introduction = self.llm.invoke([SystemMessage(introduction_prompt)]).content
        conclusion = self.llm.invoke([SystemMessage(conclusion_prompt)]).content

        final_report = f"{introduction}\n\n{content}\n\n{conclusion}"

        # Generate podcast version
        podcast_prompt = f"""You are Samantha, the host of `Tech Talk Roundtable`, moderating a roundtable discussion on {topic}.
        Create a natural conversation between you and these analysts:
        {[f"{a.name} ({a.role})" for a in analysts]}

        Base the discussion on this research:
        {content}

        Format as a podcast script with:
        [Host]: Welcome everyone...
        [Analyst Name]: Thank you for having me...
        
        Make it engaging and conversational while covering tangible key points and metrics from the research.
        
        In your outro, always end with the phrase: `Stay hungry, stay foolish.` as a tagline for the podcast"""

        podcast_version = self.llm.invoke([SystemMessage(podcast_prompt)]).content

        return {
            "final_report": final_report,
            "podcast_script": podcast_version
        }