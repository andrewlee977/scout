import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from .agent.agent import Agent
import base64

# Use relative imports for app modules
from .tools.tool_registry import get_all_tools
from app.services.services import text_to_speech
from .agent.orchestrator_agent import OrchestratorAgent

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'templates')

# Initialize FastAPI app
app = FastAPI()

# Create Jinja2Templates instance
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Add b64encode filter to Jinja2Templates
templates.env.filters["b64encode"] = lambda v: base64.b64encode(v).decode()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the main page with a simple text input form
    """
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/submit", response_class=HTMLResponse)
async def submit_text(request: Request):
    """
    Handle form submission and render the result
    """
    form_data = await request.form()
    user_text = form_data.get("user_input", "No text submitted")

    agent = Agent(get_all_tools())
    result = agent.process_message(user_text)
    # orchestrator = OrchestratorAgent(get_all_tools())
    # result = await orchestrator.process_message(user_text)

    # Only generate audio if we're not asking a follow-up question
    # audio = None
    # if not result["requires_follow_up"]:
    audio = text_to_speech(result)

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "result": result,
            "audio_content": audio
        }
    )

# Main block to run the application
def main():
    """Run the application"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()