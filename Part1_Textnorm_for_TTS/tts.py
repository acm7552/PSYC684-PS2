from google.cloud import texttospeech

import sys

def google_synthesize_text(input_text, text_file):
    """Synthesizes speech from the input string of text."""

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
    with open(f"{text_file}_output.wav", "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{text_file}_output.wav"')
        
        
def azure_synthesize_text(input_text):
    pass


def amazon_synthesize_text(input_text):
    pass


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <text_file.txt> <cloud_service>")
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
    elif cloud_service == "__":
        azure_synthesize_text(input_text, text_file)
    elif cloud_service == "____":
        amazon_synthesize_text(input_text, text_file)


if __name__=="__main__":
    main()