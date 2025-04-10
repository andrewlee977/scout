"""Contains various non-agentic services like TTS or SST"""


from openai import AsyncOpenAI
from app.config import settings
import logging
from pydub import AudioSegment
from io import BytesIO
import re
import asyncio


logger = logging.getLogger(__name__)


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
            # Count how many male voices are already assigned
            used_male_count = sum(1 for v in get_voice_for_role.speaker_voice_mapping.values() 
                                if v in male_voices)
            selected_voice = male_voices[used_male_count]
        elif gender == 'female':
            # Count how many female voices are already assigned
            used_female_count = sum(1 for v in get_voice_for_role.speaker_voice_mapping.values() 
                                  if v in female_voices)
            selected_voice = female_voices[used_female_count]
    
    if not selected_voice:
        selected_voice = 'ash'
    
    # Store the voice assignment
    get_voice_for_role.speaker_voice_mapping[role] = selected_voice
    
    # Return complete voice settings
    return {
        'voice': selected_voice,
        'instructions': voice_settings[selected_voice]['instructions']
    }


async def text_to_speech_async(text: str, voice="shimmer", instructions="") -> bytes:
   """Convert text to speech using OpenAI's API asynchronously"""
   try:
       client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
       response = await client.audio.speech.create(
           model="tts-1",
           voice=voice,
           input=text,
           instructions=instructions
       )
       return response.content
   except Exception as e:
       logger.error(f"Error in text-to-speech: {str(e)}", exc_info=True)
       raise


async def generate_podcast_audio_async(podcast: str, analysts: list) -> bytes:
   """Generate podcast audio from text script and analyst information."""
   if not podcast:
       return None
      
   try:
       # Clear the voice mapping before generating a new podcast
       if hasattr(get_voice_for_role, 'speaker_voice_mapping'):
           get_voice_for_role.speaker_voice_mapping.clear()
           
       segments = split_by_speaker(podcast, analysts)
       all_tasks = []
       segment_lengths = []  # Keep track of number of chunks per segment
      
       # Create tasks for ALL chunks across ALL segments at once
       for segment in segments:
           chunks = chunk_text(segment['content'])
           segment_lengths.append(len(chunks))
          
           for chunk in chunks:
               all_tasks.append(
                   text_to_speech_async(
                       chunk,
                       segment['voice'],
                       segment['instructions']
                   )
               )
      
       # Process everything in parallel
       all_chunk_audios = await asyncio.gather(*all_tasks)
      
       # Reconstruct the audio in correct order
       combined_audio = None
       chunk_index = 0
      
       for num_chunks in segment_lengths:
           # Get this segment's chunks
           segment_chunks = all_chunk_audios[chunk_index:chunk_index + num_chunks]
           segment_audio = None
          
           # Combine chunks for this segment
           for chunk_audio in segment_chunks:
               audio_segment = AudioSegment.from_mp3(BytesIO(chunk_audio))
               if segment_audio is None:
                   segment_audio = audio_segment
               else:
                   segment_audio += AudioSegment.silent(duration=100) + audio_segment
          
           # Add to main audio
           if combined_audio is None:
               combined_audio = segment_audio
           else:
               combined_audio += AudioSegment.silent(duration=250) + segment_audio
          
           chunk_index += num_chunks
      
       if combined_audio:
           output = BytesIO()
           combined_audio.export(output, format="mp3")
           return output.getvalue()
       return None
   except Exception as e:
       logger.error(f"Error generating podcast audio: {str(e)}")
       return None