"""Contains various non-agentic services like TTS or SST"""

from openai import OpenAI
from app.config import settings 
import logging
from pydub import AudioSegment
from io import BytesIO
import re

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

def generate_podcast_audio(podcast: str, analysts: list) -> bytes:
    """Generate podcast audio from text script and analyst information."""
    if not podcast:
        return None
        
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
                    voice=segment['voice'],
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
            return output.getvalue()
            
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None

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