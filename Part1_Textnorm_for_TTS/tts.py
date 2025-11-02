# native
import sys

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
    from elevenlabs import save

    client = ElevenLabs(
        api_key=key
    )

    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_turbo_v2_5",
        output_format="pcm_32000",
    )
    
    save(audio, f"{text_file}_ELEVEN_output.wav")
    # with open(f"{text_file}_ELEVEN_output.wav", "wb") as f:
    #     for chunk in audio:
    #         if chunk:
    #             f.write(chunk)
        

def THIRD_TTS_PLACEHOLDER():
    """Call third commercial TTS API and save output."""
    pass


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <text_file.txt> <google | eleven | third option> [optional API key]")
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
    elif cloud_service == "placeholder":
        return


if __name__=="__main__":
    main()