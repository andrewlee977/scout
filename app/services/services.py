"""Contains various non-agentic services like TTS or SST"""


from openai import OpenAI
from app.config import settings 
import logging

logger = logging.getLogger(__name__)



def text_to_speech(text: str, voice="shimmer", instructions="") -> bytes:
    """Convert text to speech using OpenAI's API"""
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            instructions=instructions
        )
        return response.content
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}", exc_info=True)
        raise