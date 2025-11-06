# Harry Hennessy, Jack Mclaughlin, Sophia Caruana, Andrew Murphy
# PSYC 684
#
# This file loads the OpenAI Whisper Tiny model, the fine tuned model, and an audio file specified in the arguments
# Once loaded, the inference from both models is printed
import torch
import argparse
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Configuration
device = torch.device("cpu")
BASE_MODEL_ID = "openai/whisper-tiny"
FINETUNED_ADAPTER_PATH = "./whisper-medical-finetuned-adapter"
MERGED_DIR = "./whisper-medical-merged-model"

# Loads an audio file and resamples it to 16000 Hz (Whisper's required rate).
def load_audio_file(file_path):
    try:
        audio, original_sr = librosa.load(file_path, sr=None)
        
        if original_sr != 16000:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
            
        print(f"Loaded audio: {file_path}")
        print(f"Original SR: {original_sr} Hz. Resampled to 16000 Hz.")
        return audio
    except FileNotFoundError:
        print(f"Error: Audio file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading audio: {e}")
        return None

# Processes audio input and generates transcription using the given model.
def transcribe(model, processor, audio_input):
    # Prepare features
    inputs = processor(
        audio_input, 
        sampling_rate=16000, 
        return_tensors="pt",
        padding=True,
    )
    
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", 
        task="transcribe"
    )
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            suppress_tokens=[],
            max_new_tokens=256,
        )

    # Decode the result
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def main(audio_file_path):
    # Model Loading
    print("\nLoading Models...")
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    
    # Load baseline model
    baseline_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID).to(device)
    print("\nBaseline Model loaded to CPU.")

    # Load fine tuned model
    finetuned_model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR).to(device)
    print(f"\nFine-Tuned Model loaded and merged to CPU.")

    # Audio Loading
    audio_input = load_audio_file(audio_file_path)
    if audio_input is None:
        return

    # Transcription
    
    print("\nRunning Baseline Transcription")
    baseline_transcription = transcribe(baseline_model, processor, audio_input)
    
    print("\nRunning Fine-Tuned Transcription")
    finetuned_transcription = transcribe(finetuned_model, processor, audio_input)

    # Output
    print(f"Input Audio File: {audio_file_path}")
    print("Baseline Transcription:\n")
    print(f"  {baseline_transcription}")
    print("\nFine-Tuned Transcription:\n")
    print(f"  {finetuned_transcription}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using baseline and fine-tuned Whisper models.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file")
    args = parser.parse_args()
    
    main(args.audio_path)