# native
import sys
import requests
import base64
import os


def google_synthesize_text(input_text, text_file):
    """Synthesizes speech from the input string of text.
    To set up authentication for Google Cloud, see 
    Application Default Credentials docs: https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment?hl=en
    """
    # must pip install "google-cloud-texttospeech" for this to work 
    from google.cloud import texttospeech
    text = input_text
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Chirp3-HD-Charon",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config,
    )

    # The response's audio_content is binary.
    with open(f"{text_file}_GOOGLE_output.wav", "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{text_file}_output.wav"')
        
def elevenlabs_synthesize_text(input_text, text_file, key):
    # pip install elevenlabs
    from elevenlabs.client import ElevenLabs
    import wave
    
    client = ElevenLabs(
        api_key=key
    )

    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="pqHfZKP75CvOlQylNhV4",
        model_id="eleven_turbo_v2_5",
        output_format="pcm_32000",
    )
    
    pcm_bytes = b"".join(audio)
    # save(audio, f"{text_file}_ELEVEN_output.mp3")
    with wave.open(f"{text_file}_ELEVEN_output.wav", "wb") as wavf:
        wavf.setnchannels(1)        # mono
        wavf.setsampwidth(2)        # 16-bit samples
        wavf.setframerate(32000)    # 16 kHz sample rate
        wavf.writeframes(pcm_bytes)
        

def inworld(input_text, text_file):
    """Call third commercial TTS API and save output."""

    url = "https://api.inworld.ai/tts/v1/voice"

    headers = {
        "Authorization": f"Basic Qlh1N1ZkSXdka1poRzhTNXZUQVRabzZnV2NJN3FuNHA6Szkwd0pRVFh6eDdST203R0hWellKdG1jSmhOaXpleExmQzlRemFEa09teklWTDk4Z2k2RGJ6ZnE5UEk2elRtUg==",
        "Content-Type": "application/json"
    }

    payload = {
        "text": input_text,
        "voiceId": "Ashley",
        "modelId": "inworld-tts-1"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    audio_content = base64.b64decode(result['audioContent'])

    with open(f"{text_file}_inworld.wav", "wb") as f:
        f.write(audio_content)


def main():
    if len(sys.argv) > 4:
        print("Usage: python script.py <text_file.txt> <google | eleven | inworld> [optional API key]")
        sys.exit(1)

    # read args
    text_file = sys.argv[1]
    cloud_service = sys.argv[2].lower()

    # read input text
    input_text = ""
    with open(text_file, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    
    # choose api
    if cloud_service == "google":
        google_synthesize_text(input_text, text_file)
    elif cloud_service == "eleven":
        elevenlabs_synthesize_text(input_text, text_file, sys.argv[3])
    elif cloud_service == "inworld":
        inworld(input_text, text_file)


if __name__=="__main__":
    main()