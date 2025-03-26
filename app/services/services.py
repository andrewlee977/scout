"""Contains various non-agentic services like TTS or SST"""


from google.cloud import texttospeech
from app.config import settings 
import logging
import os

logger = logging.getLogger(__name__)

# def text_to_speech(text: str) -> bytes:
#     try:
#         logger.info("Initializing Text-to-Speech client")
#         # Set credentials explicitly
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
#         client = texttospeech.TextToSpeechClient()
        
#         synthesis_input = texttospeech.SynthesisInput(text=text)
        
#         voice = texttospeech.VoiceSelectionParams(
#             language_code="en-US",
#             ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#         )
        
#         audio_config = texttospeech.AudioConfig(
#             audio_encoding=texttospeech.AudioEncoding.MP3
#         )
        
#         logger.info("Generating speech")
#         response = client.synthesize_speech(
#             input=synthesis_input,
#             voice=voice,
#             audio_config=audio_config
#         )
#         logger.info("Successfully generated speech")
#         return response.audio_content
#     except Exception as e:
#         logger.error(f"Error in text-to-speech: {str(e)}", exc_info=True)
#         raise


from openai import OpenAI

def text_to_speech(text: str) -> bytes:
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        return response.content
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}", exc_info=True)
        raise