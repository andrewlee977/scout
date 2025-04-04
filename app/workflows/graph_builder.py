# research_assistant/workflows/graph_builder.py

from langgraph.graph import StateGraph, START, END
from app.models.models import ResearchGraphState, InterviewState
from app.workflows.create_analysts import CreateAnalysts
from app.workflows.interview import InterviewBuilder
from app.workflows.research import ConductResearch
from langgraph.checkpoint.memory import MemorySaver


def build_interview_graph():
    interview_builder = StateGraph(InterviewState)
    
    # Create an instance of InterviewBuilder
    builder = InterviewBuilder()
    
    # Use instance methods
    interview_builder.add_node("ask_question", builder.generate_question)
    interview_builder.add_node("search_web", builder.search_web)
    interview_builder.add_node("search_wikipedia", builder.search_wikipedia)
    interview_builder.add_node("search_news", builder.search_news) #
    interview_builder.add_node("answer_question", builder.generate_answer)
    interview_builder.add_node("save_interview", builder.save_interview)
    interview_builder.add_node("write_section", builder.write_section)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("ask_question", "search_news") #
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_edge("search_news", "answer_question") #
    interview_builder.add_conditional_edges("answer_question", builder.route_messages,['ask_question','save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    
    return interview_builder


def build_research_graph():
    builder = StateGraph(ResearchGraphState)
    interview_builder = build_interview_graph()
    create_analysts = CreateAnalysts()
    conduct_research = ConductResearch()
    
    # Adding nodes
    builder.add_node("create_analysts", create_analysts.create_analysts) # current state: topic, max_analysts, analysts
    builder.add_node("human_feedback", create_analysts.human_feedback)
    builder.add_node("conduct_interview", interview_builder.compile())
    builder.add_node("write_report", conduct_research.write_report)
    builder.add_node("write_introduction", conduct_research.write_introduction)
    builder.add_node("write_conclusion", conduct_research.write_conclusion)
    builder.add_node("finalize_report", conduct_research.finalize_report)
    
    # Defining workflow logic with edges
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", conduct_research.initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)
    
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    
    return graph
