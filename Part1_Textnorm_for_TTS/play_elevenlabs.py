import sys
from elevenlabs import play

audiofile = sys.argv[1]

try:
    with open(sys.argv[1], "rb") as audio:
        play(audio)
except Exception as e:
    print(e)
    print("Usage: python play_elevenlabs.py <elevenlabs_audio_filename>")