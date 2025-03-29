import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from .workflows.agent import Agent
import base64
from app.workflows.graph_builder import build_research_graph, build_interview_graph

# Use relative imports for app modules
from .tools.tool_registry import get_all_tools
from app.services.services import text_to_speech
# from .workflows.orchestrator_agent import OrchestratorAgent

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'templates')

# Initialize FastAPI app
app = FastAPI()

# Create Jinja2Templates instance
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Add b64encode filter to Jinja2Templates
templates.env.filters["b64encode"] = lambda v: base64.b64encode(v).decode()

# Store graph in app state
@app.on_event("startup")
async def startup_event():
    app.state.graph = build_research_graph()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the main page with a simple text input form
    """
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/submit", response_class=HTMLResponse)
async def submit_text(request: Request):
    form_data = await request.form()
    user_text = form_data.get("user_input", "")

    # Use existing graph from app state
    graph = request.app.state.graph
    max_analysts = 3
    topic = user_text
    thread = {"configurable": {"thread_id": "1"}}

    # Initial run to get analysts
    graph.invoke({"topic": topic, "max_analysts": max_analysts}, thread)
    
    # Get current state to display analysts
    current_state = graph.get_state(thread)
    analysts = current_state.values.get('analysts', [])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "show_feedback_form": True,
            "analysts": analysts,
            "thread_id": thread["configurable"]["thread_id"],
            "topic": topic,
            "max_analysts": max_analysts
        }
    )

@app.post("/submit_feedback", response_class=HTMLResponse)
async def submit_feedback(request: Request):
    form_data = await request.form()
    feedback = form_data.get("feedback", "").lower()
    thread_id = form_data.get("thread_id")
    topic = form_data.get("topic")
    max_analysts = int(form_data.get("max_analysts", "3"))
    
    thread = {"configurable": {"thread_id": thread_id}}
    
    # Use existing graph from app state
    graph = request.app.state.graph
    
    # Get current state
    # current_state = graph.get_state(thread)
    print('FEEDBACK: ', feedback)
    

    # Update state with feedback
    graph.update_state(thread, {
        "human_analyst_feedback": feedback
    }, as_node="human_feedback")

    # Continue execution - graph will run initiate_all_interviews
    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50) 

    graph.update_state(thread, {"human_analyst_feedback": 
                            None}, as_node="human_feedback")
    
    # Continue
    for event in graph.stream(None, thread, stream_mode="updates"):
        print("--Node--")
        node_name = next(iter(event.keys()))
        print(node_name)

    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')

    with open('report2.md', 'w') as f:
        f.write(report)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": report
        }
    )


# Main block to run the application
def main():
    """Run the application"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()