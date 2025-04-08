import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import base64
from app.workflows.graph_builder import build_research_graph
from app.services.services import text_to_speech
import tempfile
from datetime import datetime
from io import BytesIO
from pydub import AudioSegment
import re
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi.staticfiles import StaticFiles


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
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/submit", response_class=HTMLResponse)
async def submit_text(request: Request):
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

def get_voice_for_role(role, gender=None):
    """Return voice configuration based on role and gender"""
    
    # Voice configurations
    voice_settings = {
        'shimmer': {'instructions': "Speak in a professional, broadcast style"},
        'onyx': {'instructions': "Speak with authority and gravitas"},
        'echo': {'instructions': "Speak naturally and conversationally"},
        'ash': {'instructions': "Speak clearly and precisely"},
        'nova': {'instructions': "Speak with warmth and engagement"},
        'fable': {'instructions': "Speak with energy and enthusiasm"},
        'coral': {'instructions': "Speak thoughtfully and with clarity, you choose your words carefully"}
    }
    
    # Static dictionary to store speaker-voice assignments
    if not hasattr(get_voice_for_role, 'speaker_voice_mapping'):
        get_voice_for_role.speaker_voice_mapping = {}
    
    # If this speaker already has a voice assigned, use it
    if role in get_voice_for_role.speaker_voice_mapping:
        voice = get_voice_for_role.speaker_voice_mapping[role]
        return {
            'voice': voice,
            'instructions': voice_settings[voice]['instructions']
        }
    
    # Available voices by gender
    male_voices = ['onyx', 'echo', 'ash']
    female_voices = ['nova', 'fable', 'coral']
    
    # Get the voice
    selected_voice = None
    
    if 'host' in role.lower():
        selected_voice = 'shimmer'
    elif gender:
        gender = gender.lower()
        if gender == 'male':
            used_male_voices = {v for k, v in get_voice_for_role.speaker_voice_mapping.items() 
                              if v in male_voices}
            available_voices = [v for v in male_voices if v not in used_male_voices]
            if not available_voices:
                available_voices = male_voices
            selected_voice = available_voices[0]
            
        elif gender == 'female':
            used_female_voices = {v for k, v in get_voice_for_role.speaker_voice_mapping.items() 
                                if v in female_voices}
            available_voices = [v for v in female_voices if v not in used_female_voices]
            if not available_voices:
                available_voices = female_voices
            selected_voice = available_voices[0]
    
    if not selected_voice:
        selected_voice = 'ash'
    
    # Store the voice assignment
    get_voice_for_role.speaker_voice_mapping[role] = selected_voice
    
    # Return complete voice settings
    return {
        'voice': selected_voice,
        'instructions': voice_settings[selected_voice]['instructions']
    }

def split_by_speaker(text, analysts):
    """Split podcast script into segments by speaker, including gender info"""
    segments = []
    pattern = r'(?:^|\n)\[([^:\]]+)\]:\s*([^\n]+)'
    matches = re.finditer(pattern, text)
    
    analyst_genders = {analyst.name: analyst.gender for analyst in analysts}
    
    for match in matches:
        speaker = match.group(1).strip()
        content = match.group(2).strip()
        gender = analyst_genders.get(speaker)
        
        # Get full voice configuration
        voice_config = get_voice_for_role(speaker, gender)
        
        segments.append({
            'speaker': speaker,
            'content': content,
            'gender': gender,
            'voice': voice_config['voice'],
            'instructions': voice_config['instructions']
        })
    
    return segments

def chunk_text(text, chunk_size=4000):
    """Split text into chunks at sentence boundaries"""
    if not text:
        return []
    
    sentences = text.replace('! ', '!\n').replace('? ', '?\n').replace('. ', '.\n').split('\n')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

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
        
        # Your existing audio generation code...
        audio = None
        if podcast:
            try:
                segments = split_by_speaker(podcast, analysts)
                combined_audio = None
                
                for segment in segments:
                    logger.info(f"Processing audio for: {segment['speaker']}")
                    logger.info(f"Using voice: {segment['voice']}")  # Debug print
                    
                    chunks = chunk_text(segment['content'])
                    for chunk in chunks:
                        logger.info(f"Generating audio with voice: {segment['voice']}")  # Debug print
                        chunk_audio = text_to_speech(
                            text=chunk,
                            voice=segment['voice'],  # Make sure this is passing correctly
                            instructions=segment['instructions']
                        )
                        
                        # Convert to AudioSegment
                        audio_segment = AudioSegment.from_mp3(BytesIO(chunk_audio))
                        
                        if combined_audio is None:
                            combined_audio = audio_segment
                        else:
                            combined_audio += AudioSegment.silent(duration=250) + audio_segment
                
                # Convert final audio to bytes
                if combined_audio:
                    output = BytesIO()
                    combined_audio.export(output, format="mp3")
                    audio = output.getvalue()
                    
            except Exception as e:
                logger.error(f"Error generating audio: {e}")
                audio = None

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