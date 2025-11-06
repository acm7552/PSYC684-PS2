# Harry Hennessy, Jack Mclaughlin, Sophia Caruana, Andrew Murphy
# PSYC 684
#
# This file loads the OpenAI Whisper Tiny model, the fine tuned model, and aspecified folder of audio files
# Once loaded, the inference from both models is printed for each file in the specified folder. In cases where inferences differ between baseline and fine-tuned, this information is saved to a returned CSV.

import os
import csv
import importlib.util

# import inference.py
spec = importlib.util.spec_from_file_location("inference", "inference.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

INFER_DIR = "inferences"
CSV_OUTPUT = "inference_differences.csv"

def run_batch():
    if not os.path.isdir(INFER_DIR):
        print(f"Subdirectory '{INFER_DIR}' not found.")
        return

    audio_files = [
        f for f in os.listdir(INFER_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
    ]
    if not audio_files:
        print("No audio files found in the 'inferences/' directory.")
        return

    print(f"Found {len(audio_files)} audio files in '{INFER_DIR}'.")

    # load processor and models with cache
    print("\nLoading models...")
    processor = inference.WhisperProcessor.from_pretrained(inference.BASE_MODEL_ID)

    baseline_model = inference.WhisperForConditionalGeneration.from_pretrained(
        inference.BASE_MODEL_ID
    ).to(inference.device)

    finetuned_model = inference.WhisperForConditionalGeneration.from_pretrained(
        inference.BASE_MODEL_ID
    ).to(inference.device)
    finetuned_model = inference.PeftModel.from_pretrained(
        finetuned_model, inference.FINETUNED_ADAPTER_PATH
    )
    finetuned_model = finetuned_model.merge_and_unload().to(inference.device)
    print("Models ready.\n")

    results = []

    # process all audio files
    for idx, fname in enumerate(sorted(audio_files), 1):
        fpath = os.path.join(INFER_DIR, fname)
        print("\n")
        print(f"[{idx}/{len(audio_files)}] Processing: {fname}")
        print("\n")

        try:
            audio_input = inference.load_audio_file(fpath)
            if audio_input is None:
                continue

            base_text = inference.transcribe(baseline_model, processor, audio_input)
            ft_text = inference.transcribe(finetuned_model, processor, audio_input)

            print(f"\nBaseline:\n  {base_text}")
            print(f"\nFine-Tuned:\n  {ft_text}")

            different = base_text.strip() != ft_text.strip()
            results.append({
                "file": fname,
                "baseline": base_text,
                "finetuned": ft_text,
                "different": different
            })

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # write CSV summary
    print("\nWriting summary CSV...")
    with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "baseline", "finetuned", "different"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # print differences summary
    print("\n")
    print("SUMMARY OF DIFFERENCES")
    print("\n")
    diffs = [r for r in results if r["different"]]
    if not diffs:
        print("All transcriptions identical between baseline and fine-tuned.")
    else:
        for r in diffs:
            print(f"\n• {r['file']}")
            print(f"  Baseline : {r['baseline']}")
            print(f"  Fine-Tuned: {r['finetuned']}")

    print("\n")
    print(f"Processed {len(results)} file(s). Differences found: {len(diffs)}")
    print(f"CSV saved to: {CSV_OUTPUT}")
    print("\n")


if __name__ == "__main__":
    run_batch()
