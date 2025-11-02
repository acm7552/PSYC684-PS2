import sys
from elevenlabs.client import ElevenLabs


audiofile = sys.argv[1]

try:
    with open(sys.argv[1], "rb") as audio:
        ElevenLabs.play(audio)
        
except Exception as e:
    print("Usage: python play_elevenlabs.py <elevenlabs_audio_filename>")
    raise e