import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import base64
from app.workflows.graph_builder import build_research_graph
from app.services.services import text_to_speech, generate_podcast_audio
import tempfile
from datetime import datetime
from io import BytesIO
from pydub import AudioSegment
import re
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from app.config import settings


# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'templates')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    app.state.graph = build_research_graph()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Configure static files
STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), 'app', 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create Jinja2Templates instance
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Add b64encode filter to Jinja2Templates
templates.env.filters["b64encode"] = lambda v: base64.b64encode(v).decode()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the main page with a simple text input form
    """
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": None,
        "maintenance_mode": settings.MAINTENANCE_MODE
    })

@app.post("/submit", response_class=HTMLResponse)
async def submit_text(request: Request):
    if settings.MAINTENANCE_MODE:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "maintenance_mode": True
        })

    form_data = await request.form()
    user_text = form_data.get("user_input", "")
    logger.info("User Text: %s", user_text)

    # Use existing graph from app state
    graph = request.app.state.graph
    max_analysts = 3
    topic = user_text
    thread = {"configurable": {"thread_id": str(uuid4())}}

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
    logger.info("User Feedback: %s", feedback)
    thread_id = form_data.get("thread_id")
    topic = form_data.get("topic")
    max_analysts = int(form_data.get("max_analysts", "3"))
    
    thread = {"configurable": {"thread_id": thread_id}}
    graph = request.app.state.graph

    logger.info("Gathering User feedback...")
    if feedback == "approve":
        # Continue with existing approval flow
        logger.info(f"User feedback: Approved")
        graph.update_state(thread, {"human_analyst_feedback": None}, as_node="human_feedback")
        
        # Stream the graph to continue execution
        logger.info("Continuing graph execution after approval...")
        for event in graph.stream(None, thread, stream_mode="values"):
            logger.info(f"Processing node: {event.keys()}")
        
        # Get final state after streaming
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report')
        podcast = final_state.values.get('podcast_script')
        analysts = final_state.values.get('analysts', [])
        
        # Generate audio using the service function
        audio = generate_podcast_audio(podcast, analysts) if podcast else None

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": report,
                "audio_content": audio,
                "show_feedback_form": True,
                "analysts": analysts,
                "topic": topic,
                "max_analysts": max_analysts,
                "thread_id": thread_id,
                "loading": True
            }
        )
    else:
        # Handle feedback and show new analysts
        logger.info(f"User feedback: {feedback}")
        graph.update_state(thread, {"human_analyst_feedback": feedback}, as_node="human_feedback")
        
        # Get new analysts
        new_analysts = []
        for event in graph.stream(None, thread, stream_mode="values"):
            analysts = event.get('analysts', '')
            if analysts:
                new_analysts = analysts
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "show_feedback_form": True,
                "analysts": new_analysts,
                "topic": topic,
                "max_analysts": max_analysts,
                "thread_id": thread_id,
                "previous_feedback": feedback  # Always include feedback here since we're in the else block
            }
        )

@app.get("/download-report")
async def download_report(request: Request):
    # Get thread_id from query parameters
    thread_id = request.query_params.get("thread_id")
    if not thread_id:
        logger.error("No thread_id provided for report download")
        return {"error": "No thread ID provided"}
    
    # Get the graph and thread from app state
    graph = request.app.state.graph
    thread = {"configurable": {"thread_id": thread_id}}
    
    # Get final state from graph
    final_state = graph.get_state(thread)
    final_report = final_state.values.get('final_report') if final_state else None
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        if final_report:
            # Clean the text
            final_report = final_report.replace('—', '-')
            final_report = final_report.replace('"', '"').replace('"', '"')
            final_report = final_report.replace(''', "'").replace(''', "'")
            pdf.multi_cell(0, 10, txt=final_report)
        else:
            pdf.multi_cell(0, 10, txt="No report content available.")
            logger.warning(f"Download report - No report content available for thread ID: {thread_id}")
            
        pdf.output(tmp.name)
        
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return FileResponse(
            path=tmp.name,
            filename=filename,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Security-Policy": "default-src 'self'",
                "X-Content-Type-Options": "nosniff"
            }
        )

# Main block to run the application
def main():
    """Run the application"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()